"""Unit tests for the blending steps, using synthetic images (no models needed)."""
import cv2
import numpy as np

from app import blend as B


def make_region(fake_color, crop_color, grey=False):
    crop = np.full((120, 120, 3), crop_color, np.uint8)
    fake = np.full((120, 120, 3), fake_color, np.uint8)
    if grey:
        crop[:] = crop.mean(2, keepdims=True).astype(np.uint8)
    valid = np.full((120, 120), 255, np.uint8)
    kps = np.array([[45, 50], [75, 50], [60, 65], [48, 80], [72, 80]], np.float32)
    return B.Region(0, 0, crop, fake, valid, kps, None, np.array([30, 25, 90, 105], np.float32))


def mask():
    m = np.zeros((120, 120), np.float32)
    cv2.ellipse(m, (60, 65), (25, 35), 0, 0, 360, 1, -1)
    return m


def test_tone_moves_toward_target():
    r = make_region((90, 120, 200), (60, 140, 180))
    B.match_tone(r, mask(), lighting=0.8)
    sel = mask() > 0
    assert np.abs(r.fake[sel].mean(0) - r.crop[sel].mean(0)).max() < 4


def test_lighting_transfers_shadow():
    r = make_region((150, 150, 150), (150, 150, 150))
    r.crop[:, :60] = 90  # target's left half is in shadow
    B.match_tone(r, mask(), lighting=1.0)
    left, right = r.fake[65, 45].mean(), r.fake[65, 75].mean()
    assert right - left > 25


def test_greyscale_target_gives_grey_face():
    r = make_region((90, 120, 200), (150, 150, 150), grey=True)
    B.match_tone(r, mask(), lighting=0.8)
    px = r.fake[mask() > 0].astype(int)
    assert np.abs(px[:, 0] - px[:, 2]).max() <= 3


def test_sharpness_blurs_a_too_sharp_face():
    rng = np.random.default_rng(1)
    r = make_region((128, 128, 128), (128, 128, 128))
    r.fake = rng.integers(60, 200, (120, 120, 3)).astype(np.uint8)
    r.crop = cv2.GaussianBlur(r.fake, (0, 0), 2.0)
    before = B._band_energy(r.fake, mask() > 0)
    B.match_sharpness(r, mask())
    assert B._band_energy(r.fake, mask() > 0) < before * 0.6


def test_composite_leaves_outside_untouched():
    r = make_region((0, 0, 255), (255, 0, 0))
    out = B.composite(r, mask(), B.BlendOptions())
    assert out[2, 2].tolist() == [255, 0, 0]
    assert out[65, 60].tolist() == [0, 0, 255]
