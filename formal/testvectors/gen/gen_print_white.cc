// np_pipeline.cc's to_paper_white is file-local; this translation unit includes np_pipeline.cc in place of its object
// and exposes the function to gen_print.cc unchanged.
#include "np_pipeline.cc"

void gen_to_paper_white(ParsedFilmProfile& p, double* paper_xyz) { to_paper_white(p, paper_xyz); }
