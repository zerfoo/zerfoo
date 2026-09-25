// convert-peft-adapter converts a standard PEFT LoRA adapter to GGUF so it can
// be applied by Zerfoo without Python or safetensors at inference time.
package main

import (
	"flag"
	"log"

	"github.com/zerfoo/zerfoo/training/lora"
)

func main() {
	weights := flag.String("weights", "", "PEFT adapter_model.safetensors path")
	config := flag.String("config", "", "PEFT adapter_config.json path")
	output := flag.String("output", "", "new GGUF adapter path")
	flag.Parse()
	if *weights == "" || *config == "" || *output == "" {
		log.Fatal("-weights, -config, and -output are required")
	}
	if err := lora.ConvertPEFTAdapter(*weights, *config, *output); err != nil {
		log.Fatal(err)
	}
}
