// Copyright 2021 Alpha Lam <arufa.hc@gmail.com>.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// This program consumes a build_prof.icc file generated by ArgyllCMS
// and matrix and tone curves generated in build_prof.h to create a
// ICC profile.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <lcms2.h>
#include <lcms2_plugin.h>
#include "build_prof.h"

#define COPYRIGHT "Copyright 2021 Alpha Lam (arufahc@gmail.com)."
#define MANUFACTURER "Alpha Lam (arufahc@gmail.com)"

void error_handler(cmsContext context, cmsUInt32Number error, const char* text) {
  fprintf(stderr, "Error! Code: %d, msg: %s.\n", error, text);
}

// Returns a cLUT stage from a profile, e.g. one created by ArgyllCMS.
cmsStage* read_clut_stage_16(const char* input_profile) {
  // Read profile generated by ArgyllCMS and parse the 3 stages in A2B0 tag.
  cmsHPROFILE in_profile = cmsOpenProfileFromFile(input_profile, "r");
  cmsPipeline *pipeline = cmsReadTag(in_profile, cmsSigAToB0Tag);
  cmsStage *inputstage, *clutstage, *outputstage;
  if (!cmsPipelineCheckAndRetreiveStages(
        pipeline, 3,
        cmsSigCurveSetElemType, cmsSigCLutElemType, cmsSigCurveSetElemType,
        &inputstage, &clutstage, &outputstage)) {
    fprintf(stderr, "Error! Failed to load 3 stages from A2B0 tag in build_prof.icc.\n");
    return NULL;
  }
  return clutstage;
}

// Returns a cLUT stage from a profile, e.g. one created by ArgyllCMS.
cmsStage* read_clut_stage_float(const char* input_profile) {
  cmsStage* clutstage = read_clut_stage_16(input_profile);
  // Copy the cLUT in the input profile by converting them to floats.
  _cmsStageCLutData* clut_data = (_cmsStageCLutData*)cmsStageData(clutstage);
  printf("cLUT has elements: %d\n", clut_data->nEntries);

  cmsFloat32Number* tablef = malloc(sizeof(cmsFloat32Number) * clut_data->nEntries);
  memset(tablef, 0, sizeof(cmsFloat32Number) * clut_data->nEntries);
  for (int i = 0; i < clut_data->nEntries; ++i) {
    // I can't seem to find a utility in lcms to do this. In version 4.3 specification
    // section 6.3.4.2 PCS XYZ numbers are encoded as u1Fixed15Number, which has the
    // range of [0, 1 + 32767 / 32768].
    tablef[i] = clut_data->Tab.T[i] / 32768.0f;
  }
  return cmsStageAllocCLutFloatGranular(NULL, clut_data->Params->nSamples, 3, 3, tablef);
}

cmsHPROFILE create_empty_profile() {
  cmsHPROFILE out_profile = cmsCreateRGBProfile(NULL, NULL, NULL);
  cmsSetPCS(out_profile, cmsSigXYZData);
  cmsSetDeviceClass(out_profile, cmsSigInputClass);

  cmsMLU* copyright = cmsMLUalloc(NULL, 1);
  cmsMLUsetASCII(copyright, "en", "US", COPYRIGHT);
  cmsWriteTag(out_profile, cmsSigCopyrightTag, copyright);
  cmsMLU* manufacturer = cmsMLUalloc(NULL, 1);
  cmsMLUsetASCII(manufacturer, "en", "US", MANUFACTURER);
  cmsWriteTag(out_profile, cmsSigDeviceMfgDescTag, manufacturer);

#if 0
  cmsMLU* description = cmsMLUalloc(NULL, 1);
  cmsMLUsetASCII(description, "en", "US", DESCRIPTION);
  cmsWriteTag(out_profile, cmsSigProfileDescriptionTag, description);
#endif

  cmsCIEXYZ media_white = {white_point[0], white_point[1], white_point[2]};
  cmsWriteTag(out_profile, cmsSigMediaWhitePointTag, &media_white);
  return out_profile;
}

int write_black_fallback_pipeline(cmsHPROFILE out_profile) {
  // Adds a fallback A2B0 tag which gives a black image in case D2B0 is not supported.
  cmsPipeline* black_pipeline = cmsPipelineAlloc(NULL, 3, 3);
  cmsUInt16Number black[256];
  memset(black, 0, sizeof(black));
  cmsToneCurve* black_curve[3];
  black_curve[0] = black_curve[1] = black_curve[2] = cmsBuildTabulatedToneCurve16(NULL, 256, black);
  cmsPipelineInsertStage(black_pipeline, cmsAT_BEGIN, cmsStageAllocToneCurves(NULL, 3, black_curve));

  int ret = cmsWriteTag(out_profile, cmsSigAToB0Tag, black_pipeline);
  if (!ret) {
    fprintf(stdout, "Failed to save A2B0 pipeline!\n");
  }
  return ret;
}

cmsStage* create_identity_clut_stage() {
    const cmsUInt16Number table[] = {
      0, 0, 0,
      0, 0, 0xffff,
      0, 0xffff, 0,
      0, 0xffff, 0xffff,
      0xffff, 0, 0,
      0xffff, 0, 0xffff,
      0xffff, 0xffff, 0,
      0xffff, 0xffff, 0xffff
    };
    return cmsStageAllocCLut16bit(NULL, 2, 3, 3, table);
}

// Uses A2B0 tag with mft transform that turns a crosstalk corrected linear RGB
// into inverted image.
void make_std_negative_profile_mft_clut(char* src_profile_name) {
  cmsHPROFILE out_profile = create_empty_profile();
  cmsSetProfileVersion(out_profile, 2.2);
  int ret = 0;

  // lutAToBType requires int16 vaules for the curves.
  cmsUInt16Number r_curve16[CURVE_POINTS], g_curve16[CURVE_POINTS], b_curve16[CURVE_POINTS];
  for (int i = 0; i < CURVE_POINTS; ++i) {
    r_curve16[i] = (cmsUInt16Number)(r_curve[i] * 65535);
    g_curve16[i] = (cmsUInt16Number)(g_curve[i] * 65535);
    b_curve16[i] = (cmsUInt16Number)(b_curve[i] * 65535);
  }
  cmsToneCurve* curve[3];
  curve[0] = cmsBuildTabulatedToneCurve16(NULL, CURVE_POINTS, r_curve16);
  curve[1] = cmsBuildTabulatedToneCurve16(NULL, CURVE_POINTS, g_curve16);
  curve[2] = cmsBuildTabulatedToneCurve16(NULL, CURVE_POINTS, b_curve16);

  cmsToneCurve* curvef[3];
  printf("Input curves have points: %d\n", CURVE_POINTS);
  curvef[0] = cmsBuildTabulatedToneCurveFloat(NULL, sizeof(r_curve) / sizeof(float), r_curve);
  curvef[1] = cmsBuildTabulatedToneCurveFloat(NULL, sizeof(g_curve) / sizeof(float), g_curve);
  curvef[2] = cmsBuildTabulatedToneCurveFloat(NULL, sizeof(b_curve) / sizeof(float), b_curve);
  cmsPipeline* neg_pipeline = cmsPipelineAlloc(NULL, 3, 3);
  cmsPipelineInsertStage(neg_pipeline, cmsAT_END, cmsStageAllocToneCurves(NULL, 3, curvef)); // negative tone curves.
  cmsPipelineInsertStage(neg_pipeline, cmsAT_END, read_clut_stage_16(src_profile_name));
  cmsPipelineInsertStage(neg_pipeline, cmsAT_END, cmsStageAllocToneCurves(NULL, 3, NULL)); // Identity matrix output curves.

  printf("New pipeline has stages: %d\n", cmsPipelineStageCount(neg_pipeline));

  ret = cmsWriteTag(out_profile, cmsSigAToB0Tag, neg_pipeline);
  if (!ret) {
    fprintf(stdout, "Failed to save A2B0 pipeline.\n");
  }
  cmsMD5computeID(out_profile);
  ret = cmsSaveProfileToFile(out_profile, "icc_out/std_negative_v2_clut.icc");
  if (!ret) {
    printf("Failed to save profile!\n");
  }
}

// Uses D2B0 tag with multiProcessElement Transform without crosstalk correction.
void make_std_negative_profile_mpet_mat() {
  cmsHPROFILE out_profile = create_empty_profile();
  cmsSetProfileVersion(out_profile, 4.3);
  int ret = write_black_fallback_pipeline(out_profile);

  cmsToneCurve* curvef[3];
  curvef[0] = cmsBuildTabulatedToneCurveFloat(NULL, sizeof(r_curve) / sizeof(float), r_curve);
  curvef[1] = cmsBuildTabulatedToneCurveFloat(NULL, sizeof(g_curve) / sizeof(float), g_curve);
  curvef[2] = cmsBuildTabulatedToneCurveFloat(NULL, sizeof(b_curve) / sizeof(float), b_curve);
  cmsPipeline* neg_pipeline = cmsPipelineAlloc(NULL, 3, 3);

  cmsPipelineInsertStage(neg_pipeline, cmsAT_END, cmsStageAllocToneCurves(NULL, 3, curvef)); // negative tone curves.

#if 0
  // From dctaw image b = 3 and computed by argullcms -am
  double mat[] = {0.440064, 0.520204, 0.078978,
		  0.253482, 0.826624, -0.015199,
		  0.037010, 0.266008, 0.722028};
#endif

#if 0
  // Computed by Weka using linear regression.
  double mat[] = {
    0.43828032, 0.57176434, 0.1188123,
    0.29326052, 0.82380897, 0.04753673,
    0.07779477, 0.34604537, 0.68812647
  };
  double offsets[] = {
    -3.74188932 / 100, -3.79688203 / 100, -4.18103362 / 100
  };

  cmsPipelineInsertStage(neg_pipeline, cmsAT_END, cmsStageAllocMatrix(NULL, 3, 3, mat, offsets));
#endif
  double mat[] = {
    0.550, 0.361, 0.053,
    0.259, 0.854, -0.113,
    0.011, 0.013, 0.801,
  };
  cmsPipelineInsertStage(neg_pipeline, cmsAT_END, cmsStageAllocMatrix(NULL, 3, 3, mat, NULL));

  printf("New pipeline has stages: %d\n", cmsPipelineStageCount(neg_pipeline));

  ret = cmsWriteTag(out_profile, cmsSigDToB0Tag, neg_pipeline);
  if (!ret) {
    fprintf(stdout, "Failed to save D2B0 pipeline.\n");
  }
  cmsMD5computeID(out_profile);
  ret = cmsSaveProfileToFile(out_profile, "icc_out/std_negative_mpet_mat.icc");
  if (!ret) {
    printf("Failed to save profile!\n");
  }
}

// Uses D2B0 tag with multiProcessElement Transform.
// This problem can be used with ImageMagick to convert a DCRAW linear (-o 0) file into a RGB space.
// However using such profile with Capture One will cause color shift when adjusting levels. This is
// because C1 adjusts the 'raw' values which is before the color profile is applied. When using C1
// first produce a linear image with crosstalk correction applied and then use the mpet profile
// without crosstalk correction applied.
void make_cc_negative_profile(const char* src_profile_name) {
  cmsHPROFILE out_profile = create_empty_profile();
  int ret = write_black_fallback_pipeline(out_profile);

  cmsToneCurve* curvef[3];
  curvef[0] = cmsBuildTabulatedToneCurveFloat(NULL, sizeof(r_curve) / sizeof(float), r_curve);
  curvef[1] = cmsBuildTabulatedToneCurveFloat(NULL, sizeof(g_curve) / sizeof(float), g_curve);
  curvef[2] = cmsBuildTabulatedToneCurveFloat(NULL, sizeof(b_curve) / sizeof(float), b_curve);
  cmsPipeline* neg_pipeline = cmsPipelineAlloc(NULL, 3, 3);

  cmsPipelineInsertStage(neg_pipeline, cmsAT_END, cmsStageAllocMatrix(NULL, 3, 3, crosstalk_correction_mat, NULL));
  // cmsPipelineInsertStage(neg_pipeline, cmsAT_END, create_identity_clut_stage_float());
  cmsPipelineInsertStage(neg_pipeline, cmsAT_END, cmsStageAllocToneCurves(NULL, 3, curvef)); // negative tone curves.
  cmsPipelineInsertStage(neg_pipeline, cmsAT_END, read_clut_stage_float(src_profile_name));

  printf("New pipeline has stages: %d\n", cmsPipelineStageCount(neg_pipeline));
  ret = cmsWriteTag(out_profile, cmsSigDToB0Tag, neg_pipeline);
  if (!ret) {
    fprintf(stdout, "Failed to save D2B0 pipeline.\n");
  }
  cmsMD5computeID(out_profile);
  ret = cmsSaveProfileToFile(out_profile, "icc_out/cc_negative.icc");
  if (!ret) {
    printf("Failed to save profile!\n");
  }
}

int main (int argc, char** argv) {
  cmsSetLogErrorHandler(&error_handler);
  make_std_negative_profile_mft_clut(argv[1]); // version 2.2 
  make_std_negative_profile_mpet_mat(); // version 4.3 mat
  make_cc_negative_profile(argv[1]);
  return 0;
}
