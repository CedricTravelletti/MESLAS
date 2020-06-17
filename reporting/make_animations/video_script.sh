ffmpeg -framerate 5 -loop 1 -t 25 -i gif%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
