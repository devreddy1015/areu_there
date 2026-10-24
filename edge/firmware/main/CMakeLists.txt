# The DSP is header-only (edge/dsp), so it is an include path, not a component.
# That is deliberate: no build-system coupling, and the host tests compile the
# exact same headers the firmware does.
idf_component_register(
    SRCS
        "main.cpp"
        "net.cpp"
        "tx_role.cpp"
        "rx_role.cpp"
        "telemetry.cpp"
        "csi_capture.cpp"
        "frame_auth.cpp"
    INCLUDE_DIRS
        "."
        "../../dsp"
    REQUIRES
        esp_wifi
        esp_netif
        esp_event
        nvs_flash
        esp_timer
        lwip
        esp_hw_support
        esp_system
)

# C++17 for the DSP headers. -fno-exceptions / -fno-rtti match the constraint
# the host tests build under, so a construct that works on the host but needs
# unwinding tables on target cannot slip through.
target_compile_options(${COMPONENT_LIB} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++17>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    # No floating-point contraction, for the reason in edge/dsp/test/Makefile:
    # the ESP32-S3 FPU has a fused multiply-add (madd.s), gnu++17 leaves GCC
    # at -ffp-contract=fast, and a fused a*b + c is not what the Python
    # reference computes or what the host tests verified. Without this the
    # node runs different arithmetic from the one the parity suite checks.
    -ffp-contract=off
    -Wall
    -Wextra
)
