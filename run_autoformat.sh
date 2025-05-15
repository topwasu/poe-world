#!/bin/bash
yapf -i -r --style .style.yapf .

# docformatter -i -r . \
# --exclude CausalDiscovery.jl \
# --exclude OC_Atari \
# --exclude openai-hf-interface

# isort . \
# --skip '**/CausalDiscovery.jl' \
# --skip '**/OC_Atari' \
# --skip '**/openai-hf'