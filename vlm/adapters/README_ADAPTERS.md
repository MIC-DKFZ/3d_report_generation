## WARNING: Implementation Scope and Compatibility
 
All adapters and their base classes were developed exclusively with the 
Qwen2.5VL architecture.
 
While the structure attempts to maintain a generic interface for other 
Vision-Language Models (VLM), the current implementation is **tightly 
coupled** to Qwen2.5VL's specific inputs, outputs, and internal structure.
 
Compatibility with other VLM models is NOT guaranteed and will likely 
require modifications to the adapter's interface and argument handling.