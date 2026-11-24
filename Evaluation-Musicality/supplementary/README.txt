Music-to-dance comparisons: supplementary videos
================================================

This folder is a small offline web page with the comparison videos. It needs no internet connection and no installation.


How to open it
--------------
1. Unzip the archive, keeping the folder structure.
2. Double-click index.html. It opens in your default browser; Chrome, Edge, Firefox and Safari all work.

The page opens with a "How to read" video, which explains the graphs and the beat clock. Below it are Clips 1-6.
Each clip shows three models side by side, with the song as audio: FlowerDance | MACE-Dance | Ours.
All models use the same random seed.


Controls
--------
- Each video has its own bar: play / pause, seek, speed (1x, 0.75x, 0.5x), mute and full screen.
- The tabs at the top jump to a clip.
- Starting one video pauses the others.
- Keys act on the video most in view:
    Space          play / pause
    Left / Right   2 s back / forward
    F              full screen
    M              mute / unmute

A short guide to the beat clock, BPL, the intensity graphs and IMC is at the bottom of the page.


If the videos do not play
-------------------------
Some browsers restrict pages opened from disk. In that case, use either of these:

a) Play the files directly. They are ordinary MP4 files (H.264 video, AAC audio) in videos/:
     How to read   videos/how_to_read.mp4
     Clip 1        videos/clip1.mp4
     Clip 2        videos/clip2.mp4
     Clip 3        videos/clip3.mp4
     Clip 4        videos/clip4.mp4
     Clip 5        videos/clip5.mp4
     Clip 6        videos/clip6.mp4

b) Serve the folder locally. With Python 3, run this inside the folder:
     python3 -m http.server 8000
   Then open http://localhost:8000 in a browser.


Contents
--------
index.html   the page (no external requests)
clips.js     the clip list used by the page
videos/      the seven MP4 files listed above
