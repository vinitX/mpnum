%.pdf: %.tikz template.tex
	lualatex -interaction=nonstopmode --jobname $(@:%.pdf=%) "\def\inputfile{$<} \input{template.tex}"

%.svg: %.pdf
	pdf2svg $< $@

%.png: %.pdf
	convert -background 'rgba(255,255,255,0)' -density 135 "$<" "$@"

GARBAGE := *.pdf *.svg *.aux *.log *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.run.xml *.tdo *.fmt *.out *.auxlock *.synctex\(busy\) *.toc tikz/*.log
clean:
	rm -f $(GARBAGE)
.PHONY: clean
