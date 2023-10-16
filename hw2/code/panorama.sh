# bash panorama.sh ../lake_data ../lake ../desc/desc_lake.npy 6720 7000.0
# bash panorama.sh ../mountain_data ../mountain ../desc/desc_mountain.npy 5312 4800.0

python3 cylindrical.py --input_dir $1 --h $4 --focal $5 --output_dir $2
python3 harris.py --input_dir $2 --output_dir $2 --desc_path $3
python3 stitching.py --input_dir $2 --output_dir $2 --desc_path $3
python3 crop.py --input "$2/panorama.jpg" --output "$2/panorama_crop.jpg"