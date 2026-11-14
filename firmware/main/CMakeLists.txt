idf_component_register(
    SRCS "main.c" "esp_storage.c"
         "fl_rng.c" "fl_model.c" "fl_train.c"      
         "fl_crc.c" "fl_b64.c" "fl_store.c" "fl_proto.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES driver esp_timer esp_partition esp_system esp_hw_support
)
