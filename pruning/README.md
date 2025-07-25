

## Zeroing 
```bash
python zeroing.py [base_model] [sealed_model] [zeroing_percent]
```


## Calculate BER
```bash
python extract_watermark.py [sealed_model] [non_sealed_raw_model] [output_watermark_path]
```

## Calculate p-value

```bash
python p_value.py [total_N] [ber]
```

total_N = (rank) x (rank) x (number_of_target_modules) x (number_of_layers)