add_executable(gdb_pt2 main.cc)
set_target_properties(gdb_pt2 PROPERTIES
  OUTPUT_NAME pt2
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps/chemistry_gdb_pt2"
)
sbd_configure_diag(gdb_pt2)
