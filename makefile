CXX = nvcc

CXXFLAGS += -std=c++11 -O3 \
            -Xptxas -fmad=false \
            -gencode arch=compute_75,code=sm_75 \
            -Xcompiler -fno-fast-math

CNN_SRCS = lib/cnn.cuh \
	lib/utils.cuh lib/utils.cu \
	lib/cnn_seq.cuh lib/cnn_seq.cu \
	cnn_gpu.cuh cnn_gpu.cu \
	lib/main.cu

VADD_SRCS = lib/utils.cuh lib/utils.cu lib/vadd.cu

test: cnn
	. ./params.sh; ./cnn 2>&1 | tee run.log
	@nvidia-smi >> run.log

test-seq: cnn
	@$(MAKE) --no-print-directory test SEQUENTIAL=

cnn: $(CNN_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o, $^)

test-vadd: vadd
	. ./params.sh; ./$<

vadd: $(VADD_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o, $^)

clean:
	$(RM) vadd cnn *.log

zip:
	sudo apt install -y zip
	zip -r lab3.zip run.log params.sh cnn_gpu.cu lab3-report.pdf