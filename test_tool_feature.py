# test_tool_feature.py
import sys
import os

# Set a debug flag for comprehensive inspection
# This isn't necessary for the fix, but good for debugging environmental issues
os.environ['VERTEXAI_DEBUG'] = 'true'

try:
    from vertexai.generative_models import Tool, FunctionDeclaration

    # --- DIAGNOSTIC PRINTS ---
    print("\n--- Diagnostic Information ---")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"sys.path (import search order):")
    for i, p in enumerate(sys.path):
        print(f"  {i}: {p}")
    print("-" * 30)

    print(f"FunctionDeclaration object: {FunctionDeclaration}")
    print(f"FunctionDeclaration module: {FunctionDeclaration.__module__}")
    print(f"FunctionDeclaration file location: {sys.modules[FunctionDeclaration.__module__].__file__}")
    
    print("-" * 30)
    print(f"Tool object: {Tool}")
    print(f"Tool module: {Tool.__module__}")
    print(f"Tool file location: {sys.modules[Tool.__module__].__file__}")
    print("---------------------------\n")

    # Check if a specific attribute exists before calling
    if not hasattr(FunctionDeclaration, 'type_to_schema'):
        print("DIAGNOSIS: The imported FunctionDeclaration object does NOT have 'type_to_schema' attribute.")
        print(f"This indicates the FunctionDeclaration object being used is not the one expected from vertexai.generative_models, or its definition is incomplete.")
        raise AttributeError("Diagnosed: type object 'FunctionDeclaration' has no attribute 'type_to_schema'")

    # First, test if FunctionDeclaration.type_to_schema exists (it should if Tool.from_function does)
    test_params = FunctionDeclaration.type_to_schema({"name": str})
    print(f"FunctionDeclaration.type_to_schema works. Parameters: {test_params}")

    # Now, test Tool.from_function
    my_test_tool = Tool.from_function(
        func=lambda name: f"Hello, {name}!", # Use lambda for simplicity
        name="hello_world_tool",
        description="A simple tool to say hello.",
        parameters=test_params # Use the schema we just created
    )
    print("SUCCESS: Tool.from_function is available and works!")
    print(f"Created tool: {my_test_tool}")
except AttributeError as e:
    print(f"FAILURE: AttributeError: {e}")
    print("This confirms Tool.from_function/FunctionDeclaration.type_to_schema is NOT available in your current environment.")
    print("This is highly unusual given google-cloud-aiplatform 1.94.0. Investigate PYTHONPATH, environment variables, or other shadowing modules.")
except Exception as e:
    print(f"An unexpected error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()