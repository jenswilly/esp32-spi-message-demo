#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <vector>

#include "decode_raw_mobilenet.h"
#include "esp32_spi_impl.h"
#include "spi_api.hpp"

#define MAX_DETECTIONS 16

static const char* METASTREAM = "spimetaout";
static const char* PREVIEWSTREAM = "spipreview";

extern "C" {
   void app_main();
}

static std::vector<std::string> labels = {
	"background",
	"aeroplane",
	"bicycle",
	"bird",
	"boat",
	"bottle",
	"bus",
	"car",
	"cat",
	"chair",
	"cow",
	"diningtable",
	"dog",
	"horse",
	"motorbike",
	"person",
	"pottedplant",
	"sheep",
	"sofa",
	"train",
	"tvmonitor"
};

void run_demo() {
	uint8_t req_success = 0;

	dai::SpiApi mySpiApi;
	mySpiApi.set_send_spi_impl(&esp32_send_spi);
	mySpiApi.set_recv_spi_impl(&esp32_recv_spi);

	while(1) {
		// ----------------------------------------
		// basic example of receiving data and metadata from messages.
		// ----------------------------------------
		dai::Message received_msg;

		if(mySpiApi.req_message(&received_msg, METASTREAM)) {
			// example of parsing the raw metadata
			dai::RawImgDetections det;
			mySpiApi.parse_metadata(&received_msg.raw_meta, det);

			if(det.detections.size() == 0 ) {
				// No detections
				printf(".\n");
			} else {
				printf("\n%d detections\n", det.detections.size());
				// Iterate individual detections
				for(const dai::ImgDetection& det : det.detections) {
					if(det.label <= labels.size())
						printf("label: %d, name: %s, confidence: %.2f\n", det.label, labels[det.label].c_str(), det.confidence);
					else
						printf("label: %d, confidence: %.2f\n", det.label, det.confidence);
				}
			}

			// free up resources once you're done with the message.
			// and pop the message from the queue
			mySpiApi.free_message(&received_msg);
			mySpiApi.spi_pop_messages();
		}
	}
}

// Main application
void app_main()
{
    // Init spi for the esp32
    init_esp32_spi();

    run_demo();

    // Never reached
    deinit_esp32_spi();
}
