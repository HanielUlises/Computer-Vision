CC = g++
NVCC = nvcc
CFLAGS = -Wall -Wextra -g -fopenmp `pkg-config --cflags gtk+-3.0` -I/usr/local/cuda/include
NVCCFLAGS = -g -arch=sm_86 -I/usr/local/cuda/include
SRCS_C = main.c knn.c 
OBJDIR = objects
OBJS_C = $(SRCS_C:%.c=$(OBJDIR)/%.o)
OBJS_C_TEST = $(SRCS_C_TEST:%.c=$(OBJDIR)/%.o)
OBJS_CU = $(SRCS_CU:%.cu=$(OBJDIR)/%.o)
EXEC = perceptron
TEST_EXEC = knn

all: $(EXEC) $(TEST_EXEC)

$(EXEC): $(OBJS_C) $(OBJS_CU)
	$(CC) -o $@ $^ $(LDFLAGS)

$(TEST_EXEC): $(OBJS_C_TEST)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: %.c
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cu
	mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(EXEC) $(TEST_EXEC)

.PHONY: clean
