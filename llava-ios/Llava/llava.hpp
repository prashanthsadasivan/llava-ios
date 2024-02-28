//
//  llava.hpp
//  llava-ios
//
//  Created by Prashanth Sadasivan on 2/13/24.
//

#ifndef llava_hpp
#define llava_hpp

#include "ggml.h"
#include "clip.hpp"

#define LLAVA_API __attribute__ ((visibility ("default")))

struct clip_ctx;

#ifdef __cplusplus
extern "C" {
#endif

struct llava_image_embed {
    float * embed;
    int n_image_pos;
};

/** sanity check for clip <-> llava embed size match */
LLAVA_API bool llava_validate_embed_size(const struct llama_context * ctx_llama, const struct clip_ctx * ctx_clip);

/** build an image embed from image file bytes */
LLAVA_API struct llava_image_embed * llava_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length);
/** build an image embed from a path to an image filename */
LLAVA_API struct llava_image_embed * llava_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path);
LLAVA_API void llava_image_embed_free(struct llava_image_embed * embed);
/** free an embedding made with llava_image_embed_make_* */

/** write the image represented by embed into the llama context with batch size n_batch, starting at context pos n_past. on completion, n_past points to the next position in the context after the image embed. */
LLAVA_API bool llava_eval_image_embed(struct llama_context * ctx_llama, const struct llava_image_embed * embed, int n_batch, int * n_past);


#ifdef __cplusplus
}
#endif


#endif /* llava_hpp */
