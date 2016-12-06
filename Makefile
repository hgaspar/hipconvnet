

HIP_PATH=/opt/rocm/hip
HIPCC=$(HIP_PATH)/bin/hipcc
HIPLD=$(HIP_PATH)/bin/hipcc


HIP_BLAS=$(HIP_PATH)/hcblas
HIP_BLAS_INCLUDE=$(HIP_BLAS)/include


#export HSA_PATH = /opt/rocm/hsa

#enable profiling
#HIPCC_FLAGS += -DPROFILING

GCC_VER ?= 4.8

GCC_CUR_VER = $(shell gcc -dumpversion)
GPP_CUR_VER = $(shell g++ -dumpversion)

GCC_CUR = 0
GPP_CUR = 1

ifeq ($(findstring $(GCC_VER),$(GCC_CUR_VER)),$(GCC_VER))
GCC_CUR = GCC_VER
endif

ifeq ($(findstring $(GCC_VER),$(GPP_CUR_VER)),$(GCC_VER))
GPP_CUR = GCC_VER
endif

ifeq ($(GCC_CUR), $(GPP_CUR))
    HIPCC_FLAGS += -I /usr/include/x86_64-linux-gnu -I /usr/include/x86_64-linux-gnu/c++/$(GCC_VER) -I /usr/include/c++/$(GCC_VER)
else
    $(warning )
    $(warning ***************************************************)
    $(warning *** The supported version of gcc and g++ is $(GCC_VER) ***)
    $(warning ***    Current default version of gcc is $(GCC_CUR_VER)    ***)
    $(warning ***    Current default version of g++ is $(GPP_CUR_VER)    ***)
    $(warning ***************************************************)
    $(warning )
endif
#### GCC system includes workaround ####



MODELNAME := _ConvNet

INCLUDES :=  -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I./include -I./include/common -I./include/cudaconv2 -I./include/nvmatrix

ifeq ($(HIP_PLATFORM), nvcc)
LIB := -lpthread -L$(ATLAS_LIB_PATH) -L$(CUDA_INSTALL_PATH)/lib64 -lcblas
else ifeq ($(HIP_PLATFORM), hcc)
LIB := -lpthread -L$(ATLAS_LIB_PATH) -lcblas
endif

USECUBLAS   := 1

PYTHON_VERSION=$(shell python -V 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
LIB += -lpython$(PYTHON_VERSION)

#HGSOS GENCODE_ARCH := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
#GENCODE_ARCH := -gencode=compute_20, arch=compute_20, code=\"sm_20,compute_20\"
COMMONFLAGS := -DNUMPY_INTERFACE -DMODELNAME=$(MODELNAME) -DINITNAME=init$(MODELNAME)

EXECUTABLE	:= $(MODELNAME).so

CUFILES				:= $(shell echo src/*.cu src/cudaconv2/*.cu src/nvmatrix/*.cu)
CU_DEPS				:= $(shell echo include/*.cuh include/cudaconv2/*.cuh include/nvmatrix/*.cuh)
CCFILES				:= $(shell echo src/common/*.cpp)
C_DEPS				:= $(shell echo include/common/*.h)

$(warning )
$(warning 11111111111111)
$(warning )


include common-gcc-cuda-5.0.mk
	
$(warning 222222222222222)

makedirectories:	
#	$(VERBOSE)mkdir -p $(LIBDIR)
#	$(VERBOSE)mkdir -p $(OBJDIR)/src/cudaconv2
#	$(VERBOSE)mkdir -p $(OBJDIR)/src/nvmatrix
#	$(VERBOSE)mkdir -p $(OBJDIR)/src/common
#	$(VERBOSE)mkdir -p $(TARGETDIR)
