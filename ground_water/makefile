file=main
 
all: pdf 
	# make pdf
	make bib
	make pdf
	make pdf
	make clean

out:
	if  [ -f $(file).out ] ; then cp $(file).out tmp.out; fi ;
	sed 's/BOOKMARK/dtxmark/g;' tmp.out > x.out; mv x.out tmp.out ;

pdf:
	xelatex $(file).tex

bib:
	bibtex $(file).aux

clean:
	rm -f *.nav *.snm *.toc *.vrb *.aux *.log *.out *.bbl *.blg;
	rm -f ./asset/*.aux

r:
	mupdf $(file).pdf &

x:
	mupdf-x11 $(file).pdf &

l:
	mupdf-gl $(file).pdf &

git:
	make add
	make commit
	make push

add:
	git add .

commit:
	git commit -m "update"

remote:
	git remote add origin https://gitee.com/zxpwhu/deep-surrogate-model.git

push:
	git push -u origin "master"

pull:
	git pull
