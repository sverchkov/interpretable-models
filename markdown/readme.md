Notes and presentation sources written in markdown live here.

We assume mathjax is the math renderer for the most part.
I use the Markdown+Math VSCode plugin for previewing.

## Building slides:

Simplest process for reveal.js slides is to point pandoc to a CDN by setting `revealjs-url` to https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.9.2.
(Pandoc version 2.9.2.1 does not work with reveal.js 4.0.0 and above).

Command to build revealjs slides:
```
> pandoc -t revealjs -s -o html/output.html markdown/input.html
```

Same, with mathjax (pandoc docs say mathjax doesn't work with -s, but it seems to work anyway)
```
> pandoc -t revealjs --mathjax -s -o html/output.html markdown/input.html
```