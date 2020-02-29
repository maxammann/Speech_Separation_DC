# Shuffle and select from voxceleb2 dataset:
```bash
cat vox2_max_1.csv | tail -c +2 | awk "{if (\$1) print \$1;}" | shuf | head -n 50 > vox2_max_1_m.csv
cat vox2_max_1_test.csv | tail -c +2 | awk "{if (\$2) print \$2;}" | shuf | head -n 17 | cut -c 6-
unzip -d vox2_aac_unpack vox2_aac.zip $(cat vox2_aac_unpack/vox2_max_1_f.csv vox2_aac_unpack/vox2_max_1_m.csv)

# Random unpack from ZIP:
unzip -l vox2_aac.zip  "*/" | awk '{ zipp = $4; if (gsub("\/","", $4) == 3) { print $1 zipp"*" }}' | shuf | head -n 100
```

# Merge audio files:
```bash
ffmpeg -i 8.wav -i 11.wav -filter_complex amerge 8_11.wav
```

# Calculate duration statistics:
```bash
find . -name "*.m4a" -exec echo -ne "{}\t" \; -exec ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {} \; > duration.csv
paste -sd+ vox2_aac_unpack/duration.csv | bc
cat duration.csv | awk "{print \$2}" | paste -sd+ | bc
```

# Play files from agender dataset:
```bash
aplay -f S16_LE a12635s16.raw
aprobe: ffmpeg: ffprobe -f s16le
```

# Connect to breakout
```bash
sshpass -f hsaugsburg-pass ssh -L 8000:localhost:8000 -L 6006:localhost:6006 -L 8888:localhost:8888 ammannma@breakout.hs-augsburg.de -p2222 -t tmux attach -t 0
```

# Backup
```bash
rsync -e "ssh -p 2222" -ra --progress ammannma@breakout.hs-augsburg.de:/rzhome/ammannma/*backup* .
```

# Timit convert
Convert and copy:
```bash
find -name "*.WAV" -exec ffmpeg -hide_banner -loglevel panic -ac 1 -f s16le -ar 16000 -i {} -ac 1 -acodec pcm_f32le -ar 8000 {}.riff.wav \;
find TIMIT -name '*.riff.wav' -exec cp --parents \{\} TIMIT-wav \;
```

# TEDLIUM convert
Convert:
```bash
for f in *.sph; do sox -t sph "$f" -r 8000 -t wav "${f%.*}.wav"; done
rm *.sph
```
Move to subdirs
```bash
find . -name "*.wav" -exec sh -c 'NEWDIR=`basename "$1" .wav` ; mkdir "$NEWDIR" ; mv "$1" "$NEWDIR" ' _ {} \;
```

# WSJ convert
Convert:
```bash
find . -name "*\.wv1" -o -name "*\.WV1" -exec bash -c 'NEW_PATH=${1%.*}.wav; NEW_PATH=../csr-i-wsj0-complete-wav/${NEW_PATH^^}; mkdir -p $(dirname $NEW_PATH); /fast/ammannma/speech-separation/sph2pipe_v2.5/sph2pipe -f wav $1 $NEW_PATH' _ {} \;
```
```bash
find . -name "*\.wv2" -o -name "*\.WV2" -exec bash -c 'NEW_PATH=${1%.*}.wav; NEW_PATH=../csr-i-wsj0-complete-wav-no-other-mic/${NEW_PATH^^}; mkdir -p $(dirname $NEW_PATH); /fast/ammannma/speech-separation/sph2pipe_v2.5/sph2pipe -f wav $1 $NEW_PATH' _ {} \;
```
Resample:
```bash
find csr-i-wsj0-complete-wav csr-i-wsj0-complete-wav-no-other-mic -name "*.WAV" -exec ffmpeg -hide_banner -loglevel panic -i {} -ac 1 -acodec pcm_f32le -ar 8000 {}.resampled.wav \;
```
Copy resampled:
```
rsync -r --include '*/' --include '*.resampled.wav' --exclude '*' --prune-empty-dirs csr-i-wsj0-complete-wav-no-other-mic csr-i-wsj0-complete-wav-other-mic-resampled
```