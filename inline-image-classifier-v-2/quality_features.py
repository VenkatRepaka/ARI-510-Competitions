"""Hand-made image-quality features for artifact detection (blur, exposure, haze/fog, obstruction, smears).

WHAT THIS FILE DOES
    For every image (train, validation and test) it computes 74 numbers that describe the image's *quality*:
    how sharp it is, how bright, how hazy, and whether some part of the frame differs from the rest.
    The result is one table (1177 images x 74 numbers) saved to features/quality.npz.

WHY HAND-MADE NUMBERS
    Several of the five artifact types are low-level quality problems that classic image-processing formulas
    can measure directly, without a neural network:
        blur            -> few sharp edges, little fine detail
        exposure fault  -> many pixels blown out to pure white or crushed to pure black
        lens fog        -> low contrast, washed-out colours, a milky veil
        obstruction     -> one part of the frame (usually an edge) looks different from the rest
    The neural-network features (embed_features.py) are used next to these, because these numbers alone cannot
    tell lens fog from weather fog. On their own they reach about 0.78 F1.

BACKGROUND
    Most of the five artifact types (blur, exposure, fog, contamination) are standard image-quality problems,
    with established measurements in the image-processing literature:
        sharpness / blur   focus measures used for camera autofocus and blur detection [1][2][3]
        frequency content  sharpness measured from the Fourier spectrum [4]
        edges              Canny edge detector [5]
        contrast           RMS contrast [6]; histogram entropy [7]
        haze / fog         dark channel prior [8]; the fog-density measure FADE [9] combines very similar cues
                           (dark channel, saturation, local contrast, entropy) to predict how foggy a photo is
    3x3 TILES: obstructions, drops and smears usually cover only part of the frame, so the image is also
    measured region by region, as in partial-blur detection [10] and lens-soiling detection on car cameras,
    where SoilingNet [11] classifies each cell of a grid on the image.

REFERENCES
    [1] Krotkov, E. (1988). Focusing. International Journal of Computer Vision 1(3), 223-237.
        (Tenengrad: sum of squared Sobel gradients as a focus measure.)
    [2] Pech-Pacheco, J. L., Cristobal, G., Chamorro-Martinez, J., Fernandez-Valdivia, J. (2000). Diatom
        autofocusing in brightfield microscopy: a comparative study. Proc. ICPR 2000, vol. 3, 314-317.
        (Variance of the Laplacian as a focus measure.)
    [3] Pertuz, S., Puig, D., Garcia, M. A. (2013). Analysis of focus measure operators for shape-from-focus.
        Pattern Recognition 46(5), 1415-1432. (Survey comparing Laplacian, gradient and other focus measures;
        the best single reference for the sharpness group.)
    [4] De, K., Masilamani, V. (2013). Image sharpness measure for blurred images in frequency domain.
        Procedia Engineering 64, 149-158. (Share of high-frequency energy in the spectrum as a sharpness score.)
    [5] Canny, J. (1986). A computational approach to edge detection. IEEE Trans. PAMI 8(6), 679-698.
    [6] Peli, E. (1990). Contrast in complex images. Journal of the Optical Society of America A 7(10),
        2032-2040. (RMS and local contrast.)
    [7] Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal 27,
        379-423. (Entropy.)
    [8] He, K., Sun, J., Tang, X. (2009). Single image haze removal using dark channel prior. Proc. CVPR 2009.
        (Also IEEE Trans. PAMI 33(12), 2341-2353, 2011.)
    [9] Choi, L. K., You, J., Bovik, A. C. (2015). Referenceless prediction of perceptual fog density and
        perceptual image defogging. IEEE Trans. Image Processing 24(11), 3888-3901. (FADE.)
    [10] Liu, R., Li, Z., Jia, J. (2008). Image partial blur detection and classification. Proc. CVPR 2008.
    [11] Uricar, M., Krizek, P., Sistu, G., Yogamani, S. et al. (2019). SoilingNet: Soiling detection on
        automotive surround-view cameras. Proc. IEEE ITSC 2019. arXiv:1905.01492.

HOW THE FILE IS ORGANISED (read in this order)
    1. load_rgb()     open one image upright, in RGB, shrunk to at most 1024 px
    2. stats()        25 measurements on one image (or one piece of an image)
    3. features()     run stats() on the whole image and on a 3x3 grid -> the final 74 numbers
    4. all_images()   list every image from train.csv, validation.csv and test.csv
    5. main()         loop over all images and save features/quality.npz

OUTPUT (features/quality.npz)
    ids    (1177,)     image_id of each row, so other scripts can look rows up by id
    split  (1177,)     "train", "validation" or "test"
    X      (1177, 74)  the numbers
    names  (74,)       the name of each column, e.g. "g_lap_var"

TWO WAYS OF WRITING IMAGE SIZES (easy to mix up)
    People and PIL write sizes as WIDTH x HEIGHT (columns x rows): "a 1024 x 768 image" is 1024 px wide.
    PIL:   im.size        -> (width, height)
    numpy: array.shape    -> (height, width, 3)   i.e. rows first, then columns, then the 3 colour channels
           array[r, c]    -> row r (vertical position), column c (horizontal position)
    The code always reads h and w from the array itself, so it never mixes them up.

Usage: python quality_features.py      (about 4-5 minutes for all 1177 images)
"""
import sys                     # sys.exit() at the bottom
import time                    # timing for the progress messages
from pathlib import Path       # building file paths

import cv2                     # OpenCV: the image-processing formulas (edges, blur, colour conversion)
import numpy as np             # arithmetic on whole images at once (an image is a grid of numbers)
import pandas as pd            # reading the CSV files and building the final table
from PIL import Image, ImageOps  # opening image files of any format (jpg, png, webp)

# ----------------------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------------------
ROOT = Path(__file__).parent   # the folder this script is in, so it works from any working directory
DATA = ROOT / "data"           # competition files: train.csv, validation.csv, test.csv, images/...
OUT = ROOT / "features"        # where the result is saved
LONG_SIDE = 1024               # shrink every image so its longer side is at most 1024 px.
                               # Big enough that blur is still visible (in a tiny thumbnail everything
                               # looks sharp), small enough to be fast (some photos are 6000 x 4000).
Image.MAX_IMAGE_PIXELS = None  # PIL refuses images above ~89 megapixels by default (a guard against
                               # malicious "decompression bomb" files). These are trusted competition
                               # images, so the limit is switched off.


# ----------------------------------------------------------------------------------------------------------
# 1. Loading one image
# ----------------------------------------------------------------------------------------------------------
def load_rgb(path, long_side=LONG_SIDE):
    """Open an image, rotate it upright, make it RGB, shrink it, and return a numpy array (height, width, 3)."""
    # exif_transpose: phones usually store a photo sideways and add a tag in the file's metadata (EXIF tag
    #   "Orientation", a number 1-8) saying how to rotate it for display. Photo viewers read the tag and show
    #   the photo upright. exif_transpose does the same: it reads the tag and rotates/flips the pixels
    #   accordingly (tag missing or 1 = already upright, nothing happens). This is what the host's instruction
    #   "honor EXIF orientation" in data/README.txt asks for. In this dataset 158 of 1177 images need it
    #   (156 with tag 6 = stored sideways, 1 with tag 3 = upside down, 1 with tag 8).
    #   ("transpose" is just the name Pillow gives to rotate-or-flip; it does not ignore the tag.)
    # convert("RGB"): some PNGs have a 4th transparency channel and some images are greyscale;
    #   this makes every image 3 channels (red, green, blue).
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    # thumbnail: shrink so that neither side is larger than long_side, KEEPING the aspect ratio
    #   (4032 x 3024 -> 1024 x 768). Smaller images are left as they are.
    #   LANCZOS is a high-quality resizing filter that does not add jagged edges (which would fool the
    #   sharpness measurements below).
    im.thumbnail((long_side, long_side), Image.LANCZOS)
    # numpy array of shape (height, width, 3), values 0-255 per pixel per colour
    return np.asarray(im)


# ----------------------------------------------------------------------------------------------------------
# 2. The 25 measurements on one image (or one piece of an image)
# ----------------------------------------------------------------------------------------------------------
def stats(rgb):
    """Quality statistics for one RGB uint8 array -> dict of 25 numbers.

    Groups: A. sharpness (6)   B. frequency content (2)   C. exposure (11)   D. colour and haze (6)
    """
    # --- preparation ---
    # gray: one brightness value per pixel, 0..255. Sharpness and brightness do not need colour, and one
    #   channel is 3x less work. float32 so the maths below does not overflow or round off.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # g: the same brightness scaled to 0..1 (0 = black, 0.5 = mid grey, 1 = white); easier to read.
    g = gray / 255.0
    # hsv: another way to describe colour: Hue (which colour), Saturation (how vivid, 0 = grey, 255 = vivid),
    #   Value (brightness). Only the saturation channel is used below.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    f = {}  # the results are collected here

    # ======================================================================================================
    # A. SHARPNESS: how many crisp edges does the image have? Blur (motion, focus) and fog destroy edges.
    # ======================================================================================================
    # lap_var - variance of the Laplacian, the classic one-number blur detector [2][3].
    #   The Laplacian filter gives ~0 where the image is flat or changes smoothly, and large values where the
    #   brightness changes abruptly (edges). Its variance (spread) is large for a sharp image, small for a
    #   blurry one. log1p(x) = log(1 + x) squeezes the raw range (~1 to ~10,000) into something the
    #   classifiers handle better.
    f["lap_var"] = np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var())

    # Sobel filters: how fast the brightness changes going left->right (gx) and top->bottom (gy) at each pixel.
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    # mag: the combined edge strength at each pixel (Pythagoras).
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # grad_mean: average edge strength over the whole image.
    f["grad_mean"] = np.log1p(mag.mean())
    # grad_p99: edge strength of the strongest 1% of pixels. A sharp photo of a mostly plain sky still has a
    #   few very crisp edges; this catches "sharp but simple" images that the mean would call blurry.
    f["grad_p99"] = np.log1p(np.percentile(mag, 99))
    # tenengrad: average of the SQUARED edge strength (a classic autofocus measure [1]); strong edges count much
    #   more than faint ones.
    f["tenengrad"] = np.log1p((mag ** 2).mean())
    # edge_density: Canny [5] is an edge detector that outputs a black/white map (edge / no edge); 100 and 200 are
    #   its sensitivity thresholds. The mean of a 0/1 map is the FRACTION of pixels that are edges.
    #   Blur and fog lower it.
    f["edge_density"] = (cv2.Canny(gray.astype(np.uint8), 100, 200) > 0).mean()
    # grad_aniso: is the image sharper in one direction than the other? When the camera moves sideways,
    #   vertical edges smear, so left-right changes (gx) get weak while top-bottom changes (gy) survive.
    #   ~0 = equally sharp in both directions (no blur, or focus blur which is the same in every direction);
    #   towards 1 = one direction much weaker (motion blur). + 1e-6 avoids dividing by zero on a flat image.
    f["grad_aniso"] = abs(np.abs(gx).mean() - np.abs(gy).mean()) / (np.abs(gx).mean() + np.abs(gy).mean() + 1e-6)

    # ======================================================================================================
    # B. FREQUENCY CONTENT: another view of sharpness.
    #   Any image is a mix of patterns at different scales: LOW frequencies = slow, large changes (a sky
    #   getting darker); HIGH frequencies = fine detail (grass, text, hair, sharp edges). Blur and fog remove
    #   the fine detail first, so a blurry image has little energy at high frequencies [4].
    # ======================================================================================================
    # Read from the inside out:
    #   g - g.mean()  remove the average brightness, otherwise it dominates the total and the ratios below
    #                 would mostly measure how bright the image is
    #   fft2          2-D Fast Fourier Transform: "brightness at each pixel" -> "how much of each pattern
    #                 size and direction is present"
    #   fftshift      move the low frequencies to the CENTRE, high frequencies to the edges, so that
    #                 "distance from the centre" = "how fine the pattern is"
    #   abs           the strength of each pattern (the FFT gives complex numbers; the phase is not needed)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    h, w = spec.shape            # rows, columns (same as the image)
    yy, xx = np.ogrid[:h, :w]    # row and column index grids (memory-light version)
    # r: distance of every spectrum cell from the centre, divided by h and w so that r means the same in a
    #   wide and a tall image. 0 = centre (coarsest patterns) ... ~0.7 = corners (finest patterns).
    r = np.sqrt(((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2)
    tot = spec.sum() + 1e-6      # total energy (+ 1e-6 avoids dividing by zero)
    # hf_ratio: share of the energy in the outer ring (fine detail). LOW in blurry images.
    f["hf_ratio"] = spec[r > 0.25].sum() / tot
    # mf_ratio: share in the middle ring (medium detail). Mild blur removes high frequencies but keeps
    #   medium ones; heavy blur or fog removes both. Shares (not raw amounts) make bright and dim images
    #   comparable. The cut-offs 0.1 and 0.25 are round numbers, not tuned.
    f["mf_ratio"] = spec[(r > 0.1) & (r <= 0.25)].sum() / tot

    # ======================================================================================================
    # C. EXPOSURE: too bright (overexposed, details burnt to white) or too dark (underexposed)?
    # ======================================================================================================
    # bright_mean: overall brightness (0..1). High = overexposed, low = underexposed.
    f["bright_mean"] = g.mean()
    # bright_std: how much the brightness varies over the image. Low = everything about the same brightness
    #   (fog, a washed-out photo, a finger covering the lens).
    f["bright_std"] = g.std()
    # Percentiles describe the brightness distribution in more detail: bright_p1 = the brightness only the
    #   darkest 1% of pixels are below; bright_p50 = the median; bright_p99 = near-brightest.
    #   Well exposed: p1 near 0 and p99 near 1. Foggy: both in the middle (e.g. 0.4 and 0.8).
    for p in (1, 5, 50, 95, 99):
        f[f"bright_p{p}"] = np.percentile(g, p)
    # clip_hi / clip_lo: share of pixels blown out to white (>= 250) or crushed to black (<= 5). This is the
    #   direct sign of an exposure fault: once a pixel hits the sensor's maximum (255) its detail is gone.
    #   250 and 5 (not exactly 255 and 0) leave a margin because JPEG compression nudges values slightly.
    #   (gray >= 250) is a True/False map; .mean() turns it into a fraction (0.30 = 30% of the image).
    #   Caveat: a correct photo of a white wall or a night sky also has many such pixels, so the classifier
    #   has to combine these with the other numbers.
    f["clip_hi"] = (gray >= 250).mean()
    f["clip_lo"] = (gray <= 5).mean()
    # dyn_range: gap between near-brightest and near-darkest pixels. Fog, glare and veiling compress it.
    #   Uses the 1st/99th percentiles instead of min/max so a single hot pixel or dust speck does not decide.
    f["dyn_range"] = np.percentile(g, 99) - np.percentile(g, 1)
    # entropy [7]: how spread out the brightness values are.
    #   hist = share of pixels in each of 64 brightness bins (0-3, 4-7, ...), adding up to 1.
    #   Entropy is high (max log2(64) = 6) when many brightness levels are used (a detailed, well-exposed
    #   photo) and low when most pixels pile into a few bins (blown-out white, black lens cap, flat grey fog).
    #   hist[hist > 0] skips empty bins because log(0) is undefined (they contribute 0 anyway).
    hist = np.histogram(gray, bins=64, range=(0, 256))[0] / gray.size
    f["entropy"] = -(hist[hist > 0] * np.log2(hist[hist > 0])).sum()

    # ======================================================================================================
    # D. COLOUR AND HAZE: lens fog / condensation, and partly lens contamination (smears also veil the image).
    # ======================================================================================================
    # sat_mean / sat_std: saturation (how vivid the colours are), scaled to 0..1. Fog adds a milky white veil,
    #   so colours fade towards grey and saturation drops.
    f["sat_mean"] = hsv[..., 1].mean() / 255
    f["sat_std"] = hsv[..., 1].std() / 255
    # Dark channel prior [8] (He et al., 2009, a well-known trick from dehazing research).
    #   Observation: in a CLEAR photo almost every small patch has some pixel that is very dark in at least
    #   one colour channel (a shadow, a dark object, or a vivid colour: bright red has almost no blue).
    #   Haze adds whitish light to EVERY pixel, so in a hazy photo even the darkest spots are not dark.
    #   rgb.min(axis=2)      for each pixel, the smallest of its red, green and blue values
    #   cv2.erode(.., 15x15) replace each pixel by the minimum over the 15x15 square around it
    #   / 255                scale to 0..1
    #   Result: ~0 almost everywhere for a clear image, high for a hazy/foggy one.
    dark = cv2.erode(rgb.min(axis=2), np.ones((15, 15), np.uint8)).astype(np.float32) / 255
    # dark_channel_mean: overall haze level.
    f["dark_channel_mean"] = dark.mean()
    # dark_channel_p10: is even the clearest 10% of the image hazy? Separates fog over the whole frame from
    #   a bright sky over only part of it.
    f["dark_channel_p10"] = np.percentile(dark, 10)
    # rms_contrast [6]: brightness spread divided by average brightness, i.e. contrast RELATIVE to how bright the
    #   image is (a variation of 0.1 means a lot in a dark image and little in a bright one).
    f["rms_contrast"] = g.std() / (g.mean() + 1e-6)
    # local_contrast [6][9]: how much fine detail there is in small neighbourhoods.
    #   blur          a heavily smoothed copy (Gaussian blur, radius ~8 px) = the image without its detail
    #   g - blur      what is left = the local detail (texture, small edges) at each spot
    #   square -> blur again -> square root = local standard deviation around each pixel; then averaged.
    #   Unlike bright_std (global: half sky + half ground has a big spread even if both halves are mush),
    #   this is low only when fine detail is gone everywhere, which is what fog, smears and blur do.
    blur = cv2.GaussianBlur(g, (0, 0), 8)
    f["local_contrast"] = np.sqrt(cv2.GaussianBlur((g - blur) ** 2, (0, 0), 8)).mean()
    return f  # 25 numbers: 6 sharpness + 2 frequency + 11 exposure + 6 colour/haze


# ----------------------------------------------------------------------------------------------------------
# 3. One image -> the final 74 numbers
# ----------------------------------------------------------------------------------------------------------
def features(path):
    """Whole-image measurements (25) + 3x3 grid summaries (48) + aspect ratio (1) = 74 numbers (dict)."""
    rgb = load_rgb(path)   # upright, RGB, long side <= 1024 px; numpy shape (height, width, 3)

    # --- Step 1: the 25 measurements on the WHOLE image ---
    # The prefix "g_" (global) keeps these names apart from the tile-based ones: "lap_var" -> "g_lap_var".
    glob = stats(rgb)
    out = {f"g_{k}": v for k, v in glob.items()}

    # --- Step 2: cut the image into a 3x3 grid and run the same stats() on each piece ---
    # h = number of rows (height), w = number of columns (width).
    # rgb[row_from:row_to, col_from:col_to] cuts out a rectangle (numpy: rows first, then columns).
    #   i (0, 1, 2) picks a band of ROWS    -> top / middle / bottom   (uses h)
    #   j (0, 1, 2) picks a band of COLUMNS -> left / centre / right   (uses w)
    #   // is integer division (pixel positions must be whole numbers).
    # The loop goes row by row, so the tiles are numbered:
    #       0 top-left      1 top-centre     2 top-right
    #       3 middle-left   4 CENTRE         5 middle-right
    #       6 bottom-left   7 bottom-centre  8 bottom-right
    # Example, an image 1024 wide and 768 tall (numpy shape (768, 1024, 3)):
    #   rows 0-255 / 256-511 / 512-767  x  columns 0-340 / 341-681 / 682-1023.
    # WHY TILES (see [10][11] in the header): many artifacts affect only PART of the frame (a finger in a corner, a water drop, a lens
    #   flare). Measured over the whole image they get averaged away; per tile they stay visible.
    h, w = rgb.shape[:2]
    tiles = [stats(rgb[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3]) for i in range(3) for j in range(3)]

    # --- Step 3: summarise the 9 tiles into 4 numbers per measurement ---
    # Keeping all 9 x 25 = 225 tile numbers would be too many for ~900 training images, and position-specific
    # (a finger top-left and a finger bottom-right would look like different features). So for 12 chosen
    # measurements (the ones where LOCAL differences are meaningful) we keep 4 summaries that do not care
    # exactly WHERE the problem is:
    #   _min                     the worst tile             e.g. one very blurry tile (a finger?)
    #   _max                     the best tile              e.g. the camera CAN focus, so not all blurry
    #   _std                     how uneven the tiles are   big = something local is going on
    #   _border_minus_centre     mean of the 8 border tiles minus the centre tile (index 4).
    #                            Obstructions (finger, strap, lens hood) come in from the EDGES while the
    #                            subject is usually in the middle. (A deliberately blurred background in a
    #                            CLEAN photo gives the same sign, so this is one clue among many.)
    # Example for lap_var with tile values [5.1, 5.3, 4.9, 5.2, 5.6, 5.0, 1.2, 4.8, 5.1]:
    #   min 1.2 (tile 6 bottom-left is blurry), max 5.6, std ~1.3, border_minus_centre 4.58 - 5.6 = -1.02
    for k in ("lap_var", "grad_mean", "tenengrad", "hf_ratio", "bright_mean", "clip_hi", "clip_lo",
              "sat_mean", "dark_channel_mean", "local_contrast", "edge_density", "bright_std"):
        v = np.array([t[k] for t in tiles])   # this measurement in each of the 9 tiles
        out[f"t_{k}_min"], out[f"t_{k}_max"], out[f"t_{k}_std"] = v.min(), v.max(), v.std()
        # np.delete(v, 4) = the 8 border tiles (the centre removed); v[4] = the centre tile
        out[f"t_{k}_border_minus_centre"] = np.delete(v, 4).mean() - v[4]

    # --- Step 4: aspect ratio = width / height (1.33 landscape, 0.75 portrait, 1.0 square) ---
    # Not a quality measure: a cheap hint about the image's source (phone photos are 4:3, web images vary).
    out["aspect"] = w / h

    # Total: 25 global (g_*) + 12 measurements x 4 summaries (t_*) = 48 + 1 aspect = 74 numbers
    return out


# ----------------------------------------------------------------------------------------------------------
# 4. The list of every image
# ----------------------------------------------------------------------------------------------------------
def all_images():
    """One table of all 1177 images: 823 train, then 118 validation, then 236 test.

    Columns: image_id, filename (e.g. images/train/image_0466.jpg), artifact_present (empty for test), split.
    Other scripts (embed_features.py, tile_embed.py, band_features.py) import this function, so every
    feature file lists the images in exactly the same order.
    """
    frames = []
    for split in ("train", "validation", "test"):
        df = pd.read_csv(DATA / f"{split}.csv")
        df["split"] = split            # remember which set each row came from
        frames.append(df)
    # ignore_index=True renumbers the rows 0..1176 instead of keeping three overlapping 0-based numberings.
    # Test rows have no artifact_present column, so pandas fills it with NaN; labels are not used here.
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------------------------------------
# 5. Run everything and save
# ----------------------------------------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)           # create features/ if it does not exist yet
    df = all_images()
    rows, t0 = [], time.time()
    # k counts from 1; fn is a relative path like images/train/image_0466.jpg, DATA / fn the full path.
    for k, fn in enumerate(df["filename"], 1):
        rows.append(featUSCISures(DATA / fn))   # 74 numbers for this image
        # Progress every 100 images, e.g. "100/1177  22s". flush=True makes the line appear immediately
        # even when the output goes to a log file (logs/quality.log).
        if k % 100 == 0 or k == len(df):
            print(f"{k}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    # The list of 1177 dictionaries becomes a table; the dictionary keys become the 74 column names.
    feat = pd.DataFrame(rows)
    # np.savez stores several arrays in one .npz file (a zip of numpy arrays):
    #   ids    lets baseline.py look up rows by image_id instead of trusting the row order
    #   split  train / validation / test
    #   X      the 1177 x 74 numbers; float32 halves the file size and is precise enough
    #   names  what each column means (used e.g. by review_mistakes.py to pick out g_lap_var)
    np.savez(OUT / "quality.npz", ids=df["image_id"].astype(str).values, split=df["split"].values,
             X=feat.values.astype(np.float32), names=np.array(feat.columns))
    print("saved", feat.shape)         # prints: saved (1177, 74)


# Runs main() only when the file is run directly (python quality_features.py). When another script does
# "from quality_features import all_images, stats", nothing runs; it only borrows the functions.
if __name__ == "__main__":
    sys.exit(main())
