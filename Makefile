
CC=gcc

.PHONY: test

all:
	@mkdir -p obj
	$(CC) src/*.c -dynamiclib -o obj/libprojcl.dylib -framework Accelerate -framework OpenCL -Wall -Werror

test:
	@mkdir -p obj
	$(CC) test/*.c -o obj/projcl_test -Lobj -lprojcl -Wall -Werror
	./obj/projcl_test

clean:
	rm -rf obj
