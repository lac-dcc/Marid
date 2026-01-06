add_library(Marid STATIC
  Analysis/ConstantBoundednessAnalysis.cpp
  Passes/CheckConstantBoundednessPass.cpp
)

# Public headers live in include/
target_include_directories(Marid
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Match LLVM RTTI configuration
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(Marid PRIVATE -fno-rtti)
endif()

# Link against required MLIR / LLVM libraries
target_link_libraries(Marid
  PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    LLVMSupport
)
