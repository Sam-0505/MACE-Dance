"""The ImperialDance choreographies used for metric validation, with music sources
copied from the dataset README (github.com/YunZhongNikki/ImperialDance-Dataset).

The motion arrays are not the 10 s the README claims (lengths vary, e.g. 1500-1700
frames at 100 fps), so eval_imperial.py locates each motion inside a padded music window.
k_ch2/k_ch3 (YouTube T28Av5QZtUc) and the BV1jv4y1z7Lt choreographies are not used:
those sources were unavailable as of 2026-10.

Add 'motion_start': <seconds into the source track> to an entry once the start has been
checked by eye (render_imperial_check.py); eval_imperial.py then uses it instead of searching.
"""
SUBSET = {
    'k_ch0': {'url': 'https://www.youtube.com/watch?v=WhHEQ-W3x5Y', 'start': 40, 'end': 55},
    'k_ch4': {'url': 'https://www.youtube.com/watch?v=DMOaEdw4KJM', 'start': 15, 'end': 27},
    'u_ch0': {'url': 'https://www.youtube.com/watch?v=96Xd1lzbfLk', 'start': 40, 'end': 55},
    'h_ch0': {'url': 'https://www.bilibili.com/video/BV1sa411F7JZ', 'start': 0, 'end': 15,
              'motion_start': 0.0},  # README start; motion length (15 s) matches the README range
}

# Choreographies whose source is a short clip holding the whole piece of music, so the motion
# can only start within ~1-2 s of the clip start. 'part' is the item in a multi-part Bilibili video.
SHORT_CLIPS = {
    'u_ch1': {'url': 'https://www.bilibili.com/video/BV1jv4y1z7Lt', 'part': 58, 'genre': 'Urban'},   # insomnia, 18.1 s
    'j_ch3': {'url': 'https://www.bilibili.com/video/BV1jv4y1z7Lt', 'part': 24, 'genre': 'Jazz'},    # change, 15.8 s
    'h_ch1': {'url': 'https://www.bilibili.com/video/BV1L84y1r7tk', 'genre': 'Hiphop'},              # 16.7 s
}
LEVELS = {'e': 'Expert', 'i': 'Intermediate', 'b': 'Beginner'}
MOTION_FPS = 100
