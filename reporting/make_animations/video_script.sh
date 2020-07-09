ffmpeg -framerate 5 -loop 1 -t 24 -i gif%d.png -c:v libx264 -profile:v high -crf 20 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p output.mp4
