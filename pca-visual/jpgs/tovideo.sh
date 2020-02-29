#!/bin/bash
ffmpeg \
-loop 1 -t 2 -format yuv444p -i pca0.jpg \
-loop 1 -t 2 -i pca1.jpg \
-loop 1 -t 2 -i pca2.jpg \
-loop 1 -t 2 -i pca3.jpg \
-loop 1 -t 2 -i pca4.jpg \
-loop 1 -t 2 -i pca5.jpg \
-loop 1 -t 2 -i pca6.jpg \
-loop 1 -t 2 -i pca7.jpg \
-loop 1 -t 2 -i pca8.jpg \
-loop 1 -t 2 -i pca9.jpg \
-loop 1 -t 2 -i pca10.jpg \
-filter_complex \
"[1:v]fade=d=1.0:t=in:alpha=1,setpts=PTS-STARTPTS+2/TB[v1]; \
 [2:v]fade=d=1.0:t=in:alpha=1,setpts=PTS-STARTPTS+4/TB[v2]; \
 [3:v]fade=d=1.0:t=in:alpha=1,setpts=PTS-STARTPTS+6/TB[v3]; \
 [4:v]fade=d=1.0:t=in:alpha=1,setpts=PTS-STARTPTS+8/TB[v4]; \
 [5:v]fade=d=1.0:t=in:alpha=1,setpts=PTS-STARTPTS+10/TB[v5]; \
 [6:v]fade=d=1.0:t=in:alpha=1,setpts=PTS-STARTPTS+12/TB[v6]; \
 [7:v]fade=d=1.0:t=in:alpha=1,setpts=PTS-STARTPTS+14/TB[v7]; \
 [8:v]fade=d=1.0:t=in:alpha=1,setpts=PTS-STARTPTS+16/TB[v8]; \
 [9:v]fade=d=1.0:t=in:alpha=1,setpts=PTS-STARTPTS+18/TB[v9]; \
 [10:v]fade=d=1.0:t=in:alpha=1,setpts=PTS-STARTPTS+20/TB[v10]; \
 [0][v1]overlay[o0]; \
 [o0][v2]overlay[o1]; \
 [o1][v3]overlay[o2]; \
 [o2][v4]overlay[o3]; \
 [o3][v5]overlay[o4]; \
 [o4][v6]overlay[o5]; \
 [o5][v7]overlay[o6]; \
 [o6][v8]overlay[o7]; \
 [o7][v9]overlay[o8]; \
 [o8][v10]overlay[o9]; \
 [o9]pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white[v1]" -map "[v1]" visualisation.mp4
