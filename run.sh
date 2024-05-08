# List of subclasses
subclasses=(
    "candle"
    "capsules"
    "cashew"
    "chewinggum"
    "fryum"
    "macaroni1"
    "macaroni2"
    "pcb1"
    "pcb2"
    "pcb3"
    "pcb4"
    "pipe_fryum"
)

# Loop through each subclass and launch training
for subclass in "${subclasses[@]}"; do
    accelerate launch train.py --sub_class "$subclass"
done
