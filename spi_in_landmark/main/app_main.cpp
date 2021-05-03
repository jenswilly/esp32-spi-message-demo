#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "decode_raw_mobilenet.h"
#include "esp32_spi_impl.h"

#include "spi_api.hpp"
#include "SpiPacketParser.hpp"

#define MAX_DETECTIONS 16

static const char* METASTREAM = "spimetaout";
static const char* PREVIEWSTREAM = "spipreview";

extern "C" {
   void app_main();
}

void debug_print_hex(uint8_t * data, int len){
    for(int i=0; i<len; i++){
        if(i%40==0){
            printf("\n");
        }
        printf("%02x", data[i]);
    }
    printf("\n");
}

// ----------------------------------------
// example callback for chunking up large messages.
// ----------------------------------------
uint32_t example_chunk_recv_size = 0;
void example_chunk_message(char* received_packet, uint32_t packet_size, uint32_t message_size){
    example_chunk_recv_size += packet_size;

    debug_print_hex((uint8_t*) received_packet, packet_size);

    if(example_chunk_recv_size >= message_size){
        example_chunk_recv_size = 0;
    }
}
// ----------------------------------------

void run_demo(){
    uint8_t req_success = 0;

    dai::SpiApi mySpiApi;
    mySpiApi.set_send_spi_impl(&esp32_send_spi);
    mySpiApi.set_recv_spi_impl(&esp32_recv_spi);


    // initialize some test data
    std::vector<std::uint8_t> contents(6912, 0);
    uint8_t packCount = 0;
    for(int i=0; i<6912; i++){
        if(i%252==0){
            contents[i] = packCount;
            packCount++;
        }
    }

    // create a Raw/Meta
    auto sp_send_msg = std::make_shared<dai::RawImgFrame>();
    dai::RawImgFrame* send_msg = sp_send_msg.get();

    if(send_msg==NULL){
        printf("Failed to create/get dai message. Aborting...\n");
        return;
    }

    dai::RawImgFrame::Specs send_msg_specs;
    send_msg_specs.width = 48;
    send_msg_specs.height = 48;
    send_msg_specs.stride = 48;
    send_msg_specs.bytesPP = 1;
    send_msg_specs.p1Offset = 0;
    send_msg_specs.p2Offset = 2304;
    send_msg_specs.p3Offset = 4608;

    dai::Timestamp ts;
    ts.sec = 1;
    ts.nsec = 123456789;

    send_msg->fb = send_msg_specs;
    send_msg->category = 0;
    send_msg->instanceNum = 1;
    send_msg->sequenceNum = 123;
    send_msg->ts = ts;

    // attach the frame data
    send_msg->data = contents;
    
    // set base/content
    std::vector<std::uint8_t> metadata;
    dai::DatatypeEnum datatype;
    send_msg->serialize(metadata, datatype);


    // just sending the same data over and over again for now.
    while(1) {

        if(mySpiApi.send_dai_message(sp_send_msg, "spiin")){
            // ----------------------------------------
            // example of getting large messages a chunk/packet at a time.
            // ----------------------------------------
            mySpiApi.set_chunk_packet_cb(&example_chunk_message);
            mySpiApi.chunk_message("spimeta");

            // ----------------------------------------
            // pop current message/metadata. this tells the depthai to update the info being passed back using the spi_cmds.
            // ----------------------------------------
            req_success = mySpiApi.spi_pop_messages();
        }
    }
}

//Main application
void app_main()
{

    // init spi for the esp32
    init_esp32_spi();

    run_demo();

    //Never reached.
    deinit_esp32_spi();

}
