for file in *.zip; do unzip -jn "$file" -d "./data/${file%.*}"; done
