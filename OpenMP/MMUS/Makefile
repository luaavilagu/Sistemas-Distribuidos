
LIBS = "-lm"
CFLAGS = "-fopenmp"

AMM_MMSR_SEQ.out: AMM_MMUS_OpenMP.o
	gcc AMM_MMUS_OpenMP.o -o AMM_MMUS_OpenMP.out $(LIBS) $(CFLAGS)

run: 
	./AMM_MMUS_OpenMP.out

clean: 
	-rm *.o AMM_MMUS_OpenMP.out
