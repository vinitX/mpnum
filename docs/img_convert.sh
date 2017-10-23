
convert -density 120 mpnum_logo.svg mpnum_logo_120.png
convert -density 144 mpnum_logo.svg mpnum_logo_144.png

for i in mpnum_*.svg tensors_*.svg; do
  convert -density 100 $i ${i/.svg/.png}
done

