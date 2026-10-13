#pragma once
// The two node roles. main.cpp picks one at compile time from Kconfig.

namespace saans {

// TX: broadcast ESP-NOW frames at a steady rate so an RX has something to
// measure the channel with. Does no sensing and holds no state about people.
void run_tx_role();

// RX: capture CSI, run the DSP, emit FeatureFrames to the gateway.
void run_rx_role();

}  // namespace saans
