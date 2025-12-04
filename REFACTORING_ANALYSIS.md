# Refactoring Analysis: Reusable Functions from go_melt.py for go_melt_env.py

## Summary

The `go_melt_env.py` file duplicates significant initialization and simulation logic that was refactored into reusable functions in `go_melt.py`. This document identifies what can be reused.

## Functions That Can Be Reused

### 1. `initialize_simulation_setup(solver_input)`
**Location in go_melt.py:** Lines 39-69  
**Current duplication in go_melt_env.py:** Lines 460-469 in `_initialize_simulation()`

**What it does:**
- Sets up Properties, Levels, Nonmesh
- Computes static mesh metadata (ne_nn, subcycle)
- Calculates mesh ratios (L1L2Eratio, L2L3Eratio)

**Current code in go_melt_env.py:**
```python
self.sim_properties = SetupProperties(self.solver_input.get("properties", {}))
self.sim_nonmesh = SetupNonmesh(self.solver_input.get("nonmesh", {}))
self.levels = SetupLevels(self.solver_input, self.sim_properties)
self.sim_ne_nn = getStaticNodesAndElements(self.levels)
self.sim_subcycle = getStaticSubcycle(self.sim_nonmesh)
```

**Can be replaced with:**
```python
from go_melt import initialize_simulation_setup

Properties, Levels, Nonmesh, ne_nn, subcycle, L1L2Eratio, L2L3Eratio = initialize_simulation_setup(self.solver_input)
self.sim_properties = Properties
self.sim_nonmesh = Nonmesh
self.levels = Levels
self.sim_ne_nn = ne_nn
self.sim_subcycle = subcycle
# Store ratios for later use
self.sim_L1L2Eratio = L1L2Eratio
self.sim_L2L3Eratio = L2L3Eratio
```

**Benefits:**
- Eliminates code duplication
- Ensures consistency between main simulation and environment
- Reduces maintenance burden

---

### 2. `initialize_interpolation(Levels)`
**Location in go_melt.py:** Lines 255-273  
**Current duplication in go_melt_env.py:** Lines 472-474 in `_initialize_simulation()`

**What it does:**
- Creates interpolation matrices between levels (L1L2Interp, L2L3Interp)

**Current code in go_melt_env.py:**
```python
L1L2Interp = interpolatePointsMatrix(self.levels[1], self.levels[2]["node_coords"])
L2L3Interp = interpolatePointsMatrix(self.levels[2], self.levels[3]["node_coords"])
self.sim_linterp = [L1L2Interp, L2L3Interp]
```

**Can be replaced with:**
```python
from go_melt import initialize_interpolation

self.sim_linterp = initialize_interpolation(self.levels)
```

**Benefits:**
- Simpler code
- Consistent with main simulation

---

### 3. Mesh Ratio Calculations in `_run_simulation_step()`
**Location in go_melt.py:** Lines 61-67 (in `initialize_simulation_setup`)  
**Current duplication in go_melt_env.py:** Lines 559-565 in `_run_simulation_step()`

**What it does:**
- Calculates L1L2Eratio and L2L3Eratio for mesh movement

**Current code in go_melt_env.py:**
```python
L1L2Eratio = [
    int(jnp.round(self.levels[1]["h"][i] / self.levels[2]["h"][i])) for i in range(2)
] + [int(jnp.round(self.sim_properties["layer_height"] / self.levels[2]["h"][2]))]

L2L3Eratio = [
    int(jnp.round(self.levels[2]["h"][i] / self.levels[3]["h"][i])) for i in range(3)
]
```

**Can be replaced with:**
- Store ratios during initialization (from `initialize_simulation_setup`)
- Reuse stored ratios instead of recalculating

**Benefits:**
- Avoids redundant calculations
- Ensures ratios are consistent

---

### 4. `handle_layer_change()` (Partial)
**Location in go_melt.py:** Lines 440-546  
**Current duplication in go_melt_env.py:** Lines 551-554 in `_run_simulation_step()`

**What it does:**
- Handles layer change logic including:
  - Finding matching z-layer in Level 1
  - Updating Level 1 state
  - Updating interpolation matrices
  - Shifting Level 0 data
  - Updating accumulated time

**Current code in go_melt_env.py:**
```python
# Handle layer changes if needed
if laser_pos_jax[2] != self.sim_laser_prev_z:
    # Layer change logic would go here
    # For now, just update the previous z
    self.sim_laser_prev_z = laser_pos_jax[2]
```

**Note:** The environment currently has simplified layer change handling. The full `handle_layer_change()` function could be used, but may need adaptation since:
- Environment doesn't track `accum_time` or `max_accum_time` (unless needed)
- Environment doesn't save checkpoints
- Environment may not need all the Level 0 shifting logic

**Potential usage:**
```python
from go_melt import handle_layer_change

if laser_pos_jax[2] != self.sim_laser_prev_z:
    # Use simplified version or full function
    # Need to adapt return values to environment's needs
    LInterp, tmp_ne_nn, laser_prev_z, force_move, move_vert, wait_inc, accum_time, max_accum_time = handle_layer_change(
        laser_pos_jax, self.sim_laser_prev_z, self.levels, self.sim_properties,
        self.sim_linterp, self.sim_nonmesh, False, False, None, None
    )
    self.sim_linterp = LInterp
    self.sim_tmp_ne_nn = tmp_ne_nn
    self.sim_laser_prev_z = laser_prev_z
```

**Benefits:**
- Proper layer change handling
- Consistent with main simulation
- Handles interpolation matrix updates automatically

---

## Recommended Refactoring Steps

### Step 1: Import reusable functions
Add to imports in `go_melt_env.py`:
```python
from go_melt import (
    initialize_simulation_setup,
    initialize_interpolation,
    handle_layer_change
)
```

### Step 2: Refactor `_initialize_simulation()`
Replace manual setup with function calls:
```python
def _initialize_simulation(self):
    """Initialize the real GO-MELT simulation state."""
    if not HAS_GO_MELT:
        print("Warning: GO-MELT simulation functions not available. Using mock simulation.")
        return False
    
    try:
        # Use reusable initialization function
        Properties, Levels, Nonmesh, ne_nn, subcycle, L1L2Eratio, L2L3Eratio = initialize_simulation_setup(self.solver_input)
        
        self.sim_properties = Properties
        self.sim_nonmesh = Nonmesh
        self.levels = Levels
        self.sim_ne_nn = ne_nn
        self.sim_subcycle = subcycle
        
        # Store mesh ratios for later use
        self.sim_L1L2Eratio = L1L2Eratio
        self.sim_L2L3Eratio = L2L3Eratio
        
        # Use reusable interpolation function
        self.sim_linterp = initialize_interpolation(self.levels)
        
        # Initialize Shapes (will be set by moveEverything)
        self.sim_shapes = None
        
        # Initialize substrate (needs getSubstrateNodes from computeFunctions)
        from computeFunctions import getSubstrateNodes
        substrate_nodes = getSubstrateNodes(self.levels)
        self.sim_substrate = substrate_nodes
        
        # Initialize tracking variables
        self.sim_laser_prev_z = float("inf")
        self.sim_move_hist = [jnp.array(0), jnp.array(0), jnp.array(0)]
        self.sim_time_inc = 0
        
        # Toolpath file handling (keep existing logic)
        # ... (rest of toolpath file logic)
        
        print("✓ GO-MELT simulation initialized successfully")
        return True
        
    except Exception as e:
        print(f"Warning: Failed to initialize GO-MELT simulation: {e}")
        import traceback
        traceback.print_exc()
        return False
```

### Step 3: Refactor `_run_simulation_step()`
Use stored mesh ratios and potentially `handle_layer_change()`:
```python
def _run_simulation_step(self, laser_position: np.ndarray, laser_power: float) -> Optional[float]:
    """Run a single step of the GO-MELT simulation."""
    if not HAS_GO_MELT or self.levels is None:
        return None
    
    try:
        # Convert laser position to JAX array format
        dt = self.sim_nonmesh.get("timestep_L3", 1e-5)
        laser_pos_jax = jnp.array([
            float(laser_position[0]),  # x
            float(laser_position[1]),  # y
            float(laser_position[2]),  # z
            1.0,  # jump
            1.0,  # dwell
            dt,   # timestep
            laser_power  # power
        ], dtype=jnp.float32)
        
        # Handle layer changes using reusable function
        if laser_pos_jax[2] != self.sim_laser_prev_z:
            # Use handle_layer_change for proper layer handling
            # Note: adapt parameters based on environment needs
            self.sim_linterp, self.sim_tmp_ne_nn, self.sim_laser_prev_z, _, _, _, _, _ = handle_layer_change(
                laser_pos_jax, self.sim_laser_prev_z, self.levels, self.sim_properties,
                self.sim_linterp, self.sim_nonmesh, False, False, None, None
            )
        
        # Move meshes if needed (use stored ratios)
        if self.sim_shapes is None or laser_pos_jax[2] != self.sim_laser_prev_z:
            laser_start = jnp.array([
                float(laser_position[0]),
                float(laser_position[1]),
                float(laser_position[2]),
                0.0, 0.0, 0.0, 0.0
            ], dtype=jnp.float32)
            
            # Use stored mesh ratios instead of recalculating
            (self.levels, self.sim_shapes, self.sim_linterp, self.sim_move_hist) = moveEverything(
                laser_pos_jax,
                laser_start,
                self.levels,
                self.sim_move_hist,
                self.sim_linterp,
                self.sim_L1L2Eratio,  # Use stored ratio
                self.sim_L2L3Eratio,  # Use stored ratio
                self.sim_properties["layer_height"],
            )
            
            # Update tmp_ne_nn
            self.sim_tmp_ne_nn = calcStaticTmpNodesAndElements(self.levels, laser_pos_jax)
        
        # Run simulation step
        self.levels, _ = stepGOMELT(
            self.levels,
            self.sim_ne_nn,
            self.sim_tmp_ne_nn,
            self.sim_shapes,
            self.sim_linterp,
            laser_pos_jax[:3],
            self.sim_properties,
            dt,
            laser_power,
            self.sim_substrate
        )
        
        # Get maximum temperature
        if get_max_temp_L3 is not None:
            max_temp = float(get_max_temp_L3(self.levels))
        else:
            max_temp = float(jnp.max(self.levels[3]["T0"]))
        
        self.sim_time_inc += 1
        return max_temp
        
    except Exception as e:
        print(f"Warning: Simulation step failed: {e}")
        import traceback
        traceback.print_exc()
        return None
```

## Additional Considerations

### Functions NOT Recommended for Direct Reuse

1. **`initialize_power_controller()`** - Environment handles power control differently (through actions)
2. **`setup_save_path()`** - Environment doesn't need timestamped save paths
3. **`initialize_toolpath()`** - Environment has its own toolpath generation logic
4. **`initialize_time_tracking()`** - Environment tracks different state variables
5. **`load_checkpoint()`** - Environment doesn't use checkpoints
6. **`warmup_jax_compilation()`** - Could be useful but environment may not need it
7. **`finalize_simulation()`** - Environment doesn't save final results

### Potential New Helper Function

Consider creating a lightweight version of `handle_layer_change()` specifically for environments that don't need all the checkpoint/accumulation logic:

```python
def handle_layer_change_simple(laser_pos, laser_prev_z, Levels, Properties, LInterp):
    """
    Simplified layer change handler for environments.
    Only updates interpolation matrices, no checkpoint/accumulation logic.
    """
    if laser_pos[2] != laser_prev_z:
        # Update interpolation matrices
        L1L2Interp = interpolatePointsMatrix(Levels[1], Levels[2]["node_coords"])
        L2L3Interp = interpolatePointsMatrix(Levels[2], Levels[3]["node_coords"])
        LInterp = [L1L2Interp, L2L3Interp]
        tmp_ne_nn = calcStaticTmpNodesAndElements(Levels, laser_pos)
        return LInterp, tmp_ne_nn, laser_pos[2]
    else:
        tmp_ne_nn = calcStaticTmpNodesAndElements(Levels, laser_pos)
        return LInterp, tmp_ne_nn, laser_prev_z
```

## Summary

**High Priority Reuse:**
1. ✅ `initialize_simulation_setup()` - Eliminates significant duplication
2. ✅ `initialize_interpolation()` - Simple, direct replacement

**Medium Priority Reuse:**
3. ⚠️ `handle_layer_change()` - Useful but may need adaptation or simplified version

**Low Priority:**
4. Store mesh ratios during initialization to avoid recalculation

**Estimated Code Reduction:** ~30-40 lines of duplicated code can be eliminated.

