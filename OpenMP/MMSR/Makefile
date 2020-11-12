
LIBS = "-lm"
CFLAGS = "-fopenmp"

AMM_MMSR_SEQ.out: AMM_MMSR_OpenMP.o
	gcc AMM_MMSR_OpenMP.o -o AMM_MMSR_OpenMP.out $(LIBS) $(CFLAGS)

run: 
	./AMM_MMSR_OpenMP.out

clean: 
	-rm *.o AMM_MMSR_OpenMP.out
