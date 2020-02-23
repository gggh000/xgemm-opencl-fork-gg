 
#binary name 
EXEC = xgemmStandaloneTest

#folder
SRC_DIR = .
OBJ_DIR = build


#/home/fpadmin/benjamin/clBLAS/src/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CLBLAS_ROOT/build/library

#sources
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(addprefix $(OBJ_DIR)/,$(notdir $(SOURCES:.cpp=.o)))
INCLUDE = -I${AMDAPPSDKROOT}/include -I. -I${HOME}/OpenBlas/include

#param
CC = g++
DEBUG = -g
WARN_FLAGS = -W -Wall -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar\
		-Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type\
		-Wno-unused-function
OPT_FLAGS = -fno-strict-aliasing -O3

CCFLAGSDEBUG = $(DEBUG) $(WARN_FLAGS) $(INCLUDES) $(OPT_FLAGS) $(INCLUDE)
CCFLAGS = $(WARN_FLAGS) $(INCLUDES) $(OPT_FLAGS) $(INCLUDE)


#link param
LIBS_PATH = -L/usr/lib64   -L${HOME}/OpenBlas/lib -L/${AMDAPPSDKROOT}/lib/x86_64
LIBS =    -lrt    -lOpenCL -lpthread -lopenblas 
LDFLAGS = $(LIBS_PATH) $(LIBS)


.SILENT:
.PRECIOUS :%.o

all: init $(EXEC)
	@echo "Build Successful"

init:
	@echo "Building $(EXEC)"
	mkdir -p $(OBJ_DIR)

$(EXEC): $(OBJECTS)
	@echo "$(CC) $(OBJ_DIR)/*.o -o $(EXEC) $(LDFLAGS)"
	$(CC) $(OBJECTS) -o $(EXEC) $(LDFLAGS)


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(OBJ_DIR)
	@echo " $(CC) $(CCFLAGS) -c $<"
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/*.o $(EXEC) 
	mkdir -p $(OBJ_DIR)
	rmdir --ignore-fail-on-non-empty -p $(OBJ_DIR)
