package msmarco

type MsMarcoRecord struct {
	Answers           []string      `parquet:"answers,list"`
	Passages          PassagesGroup `parquet:"passages"`
	Query             string        `parquet:"query"`
	QueryID           int32         `parquet:"query_id"`
	QueryType         string        `parquet:"query_type"`
	WellFormedAnswers []string      `parquet:"wellFormedAnswers,list"`
}

type PassagesGroup struct {
	IsSelected  []int32  `parquet:"is_selected,list"`
	PassageText []string `parquet:"passage_text,list"`
	URL         []string `parquet:"url,list"`
}
