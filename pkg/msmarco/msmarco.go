package msmarco

const ID = "microsoft/ms_marco"

//go:generate go tool generate_dataset_structs -dataset microsoft/ms_marco -config v2.1 -output gen_schema.go -package msmarco
