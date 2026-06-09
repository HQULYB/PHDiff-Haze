LATEX_ENGINE ?= tectonic
TECTONIC_CACHE_DIR ?= /tmp/tectonic-cache
TARGET = main

.PHONY: all clean

all:
ifeq ($(LATEX_ENGINE),tectonic)
	TECTONIC_CACHE_DIR=$(TECTONIC_CACHE_DIR) tectonic $(TARGET).tex
else
	$(LATEX_ENGINE) -interaction=nonstopmode $(TARGET).tex
	$(LATEX_ENGINE) -interaction=nonstopmode $(TARGET).tex
endif

clean:
	rm -f $(TARGET).aux $(TARGET).log $(TARGET).out $(TARGET).toc $(TARGET).fls $(TARGET).fdb_latexmk $(TARGET).synctex.gz
