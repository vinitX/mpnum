
convert -density 120 tensors_logo.svg tensors_logo_120.png 
convert -density 144 tensors_logo.svg tensors_logo_144.png 

for i in tensors_*; do
  convert -density 100 $i ${i/.svg/.png}
done

