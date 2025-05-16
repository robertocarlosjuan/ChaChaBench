# omnig-control/main.py
import traceback

# Use relative imports
from .config import SceneConfiguration
from .controller import SceneCameraController

def main():
    """Main execution function."""
    print("--- Starting Scene Camera Control ---")
    config = SceneConfiguration()
    controller = SceneCameraController(config)

    try:
        # Perform setup (environment, map, pose, recording)
        if controller.setup():
            # Run the simulation loop
            controller.run_simulation()
        else:
            print("Setup failed. Exiting.")

    except Exception as e:
        print(f"\n--- An unexpected error occurred during execution: ---")
        print(e)
        traceback.print_exc()
        print("------------------------------------------------------")
    finally:
        # Ensure cleanup happens regardless of success or failure
        controller.cleanup()
    print("--- Scene Camera Control Finished ---")

if __name__ == "__main__":
    main()