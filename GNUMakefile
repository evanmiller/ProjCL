
CC=gcc

.PHONY: test

all:
	@mkdir -p obj
	$(CC) src/*.c -dynamiclib -o obj/libprojcl.dylib -framework Accelerate -framework OpenCL -Wall -Werror

test:
	@mkdir -p obj
	$(CC) test/*.c -o obj/projcl_test -Lobj -lprojcl -Wall -Werror
	./obj/projcl_test

test_proj4:
	@mkdir -p obj
	$(CC) test/*.c -o obj/projcl_test_proj4 -DHAVE_PROJ4 -Lobj -lprojcl -lproj -Wall -Werror
	./obj/projcl_test_proj4

clean:
	rm -rf obj
