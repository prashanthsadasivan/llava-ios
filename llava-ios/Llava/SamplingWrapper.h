#import <Foundation/Foundation.h>

@interface SamplingWrapper : NSObject

// Initializers
- (instancetype)initWithLlamaCtx:(struct llama_context*)llamaCtx;
- (void)resetSamplingContext;
- (void)freeSamplingContext;


// Additional Functionalities
- (NSArray<NSNumber *> *)tokenizeText:(NSString *)text addSpecial:(BOOL)addSpecial parseSpecial:(BOOL)parseSpecial;
- (NSString *)tokenToPieceWithToken:(NSNumber *)token;
- (BOOL)evaluateTokens:(NSArray<NSNumber *> *)tokens batchSize:(NSInteger)batchSize;
- (BOOL)evaluateString:(NSString *)string batchSize:(NSInteger)batchSize addBos:(BOOL)addBos;
- (BOOL) embedImage:(UInt8*)image withSize:(int)size clipContext:(struct clip_ctx*)clip_ctx;
- (NSString *)sampleAndEvaluate;

@end
