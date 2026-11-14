CC ?= cc
CFLAGS := -std=gnu11 -Wall -Wextra -Werror -O2 -I../main
SRC := ../main/fl_rng.c ../main/fl_model.c ../main/fl_train.c ../main/fl_crc.c \
       ../main/fl_b64.c ../main/fl_store.c ../main/fl_proto.c
.PHONY: all clean
all: proto_sim
proto_sim: proto_sim.c $(SRC)
	$(CC) $(CFLAGS) -o $@ proto_sim.c $(SRC) -lm
clean:
	rm -f proto_sim
