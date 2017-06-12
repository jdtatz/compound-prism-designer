SOURCES := $(shell find "." -name '*.f')

all: fitpack.so fitpack.dll

fitpack.so: $(SOURCES)
	gfortran -shared -fPIC -o fitpack.so *.f

fitpack.dll: $(SOURCES)
	x86_64-w64-mingw32-gfortran -shared -fPIC -o fitpack.dll *.f

clean:
	-rm fitpack.so fitpack.dll
