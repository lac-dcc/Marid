add_executable(marid-opt
  marid-opt.cpp
)

# RTTI handling (match LLVM build)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(marid-opt PRIVATE -fno-rtti)
endif()

target_link_libraries(marid-opt
  PRIVATE
    Marid

    # Core MLIR
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRSupport

    # Dialects you USE
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect

    # LLVM
    LLVMSupport
)

set_target_properties(marid-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
