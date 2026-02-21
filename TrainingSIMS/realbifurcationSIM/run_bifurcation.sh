#!/usr/bin/env bash
set -e

# ===========================================================
# BIFURCATION BATCH SIMULATION - MULTI-MESH
# Reynolds Numbers: 50 - 800 (variable steps)
# Multiple mesh geometries (angle variations)
# ===========================================================

# ===================== CONFIGURATION =====================
DIAMETER=0.017638075       # Validated hydraulic diameter [m]
RHO=998.2                  # Water density [kg/m³]
MU=0.001003                # Water dynamic viscosity [Pa·s]
TEMPLATE="bifurcation_template.jou"
CONVERGENCE_THRESHOLD=1e-4  # Relaxed for continuity

# Mesh files - UPDATE THESE WITH YOUR ACTUAL FILENAMES
MESHES=(
    "coronary_extracted_vessel.msh"
)

# Reynolds number range
RES=(50 75 100 125 150 175 200 225 250 275 300 350 400 450 500 600 700 800)


# ===================== FUNCTIONS =====================

# Dynamic iteration calculator
get_iterations() {
    local re=$1
    if   (( re <= 500 )); then
        echo 1000
    elif (( re <= 1000 )); then
        echo 1500
    elif (( re <= 1500 )); then
        echo 2000
    else
        echo 2500
    fi
}

# Check convergence from console.log
check_convergence() {
    local logfile=$1
    local max_iters=$2
    
    local last_line=$(grep -E "^\s*[0-9]+\s+[0-9]" "$logfile" | tail -1)
    
    if [ -z "$last_line" ]; then
        echo "0,N/A,PARSE_ERROR"
        return 1
    fi
    
    local actual_iters=$(echo "$last_line" | awk '{print $1}')
    local continuity=$(echo "$last_line" | awk '{print $2}')
    
    local converged=$(python -c "
threshold = ${CONVERGENCE_THRESHOLD}
continuity = float('${continuity}')
if continuity < threshold:
    print('CONVERGED')
else:
    print('NOT_CONVERGED')
")
    
    echo "${actual_iters},${continuity},${converged}"
    
    if [ "$converged" == "CONVERGED" ]; then
        return 0
    else
        return 1
    fi
}

# ===================== MAIN =====================
echo "=========================================="
echo "   BIFURCATION MULTI-MESH BATCH SIM      "
echo "=========================================="
echo "Diameter:  $DIAMETER m"
echo "Density:   $RHO kg/m³"
echo "Viscosity: $MU Pa·s"
echo "Re List:   ${RES[*]}"
echo "Convergence: $CONVERGENCE_THRESHOLD"
echo "Meshes:    ${#MESHES[@]}"
echo "=========================================="

# Calculate kinematic viscosity
NU=$(python -c "print(${MU}/${RHO})")
echo "Kinematic Viscosity: $NU m²/s"
echo ""

TOTAL_RE=${#RES[@]}
TOTAL_MESHES=${#MESHES[@]}
TOTAL_CASES=$((TOTAL_RE * TOTAL_MESHES))
CURRENT_CASE=0
FAILED_CASES=()
NOT_CONVERGED_CASES=()

# Create results directory
mkdir -p results

# Log file for summary
LOGFILE="results/batch_summary.log"
echo "Batch started: $(date)" > "$LOGFILE"
echo "Mesh,Re,Velocity,MaxIters,ActualIters,FinalResidual,Status,Time" >> "$LOGFILE"

# ===================== MESH LOOP =====================
for MESH in "${MESHES[@]}"; do
    MESH_NAME=$(basename "$MESH")
    MESH_NAME=${MESH_NAME%.msh.h5}
    MESH_NAME=${MESH_NAME%.msh}

    
    echo ""
    echo "############################################"
    echo "  MESH: $MESH_NAME"
    echo "############################################"
    echo ""
    
    # Check if mesh file exists
    if [ ! -f "$MESH" ]; then
        echo "  ✗ ERROR: Mesh file '$MESH' not found! Skipping..."
        continue
    fi
    
    # ===================== REYNOLDS LOOP =====================
    for Re in "${RES[@]}"; do
        CURRENT_CASE=$((CURRENT_CASE + 1))
        START_TIME=$(date +%s)
        
        # Calculate velocity
        VELOCITY=$(python -c "print(${Re} * ${NU} / ${DIAMETER})")
        
        # Get dynamic iteration count
        ITERS=$(get_iterations $Re)
        
        # Create output directory
        mkdir -p "results/${MESH_NAME}/Re${Re}"
        
        # Generate journal file
        sed -e "s|MESH_FILE|${MESH}|g" \
            -e "s|MESH_NAME|${MESH_NAME}|g" \
            -e "s|VALUE_VELOCITY|${VELOCITY}|g" \
            -e "s|VALUE_RE|${Re}|g" \
            -e "s|VALUE_ITERS|${ITERS}|g" \
            "$TEMPLATE" > "run_${MESH_NAME}_Re${Re}.jou"
        
        echo "────────────────────────────────────────"
        echo "[$CURRENT_CASE/$TOTAL_CASES] $MESH_NAME | Re = $Re"
        echo "  Velocity:   $VELOCITY m/s"
        echo "  Max Iters:  $ITERS"
        echo "────────────────────────────────────────"
        
        # Run Fluent with live output
        if fluent 3ddp -g -t4 -i "run_${MESH_NAME}_Re${Re}.jou" 2>&1 | tee "results/${MESH_NAME}/Re${Re}/console.log"; then
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            
            # Check convergence
            CONV_RESULT=$(check_convergence "results/${MESH_NAME}/Re${Re}/console.log" $ITERS)
            ACTUAL_ITERS=$(echo "$CONV_RESULT" | cut -d',' -f1)
            FINAL_RESIDUAL=$(echo "$CONV_RESULT" | cut -d',' -f2)
            CONV_STATUS=$(echo "$CONV_RESULT" | cut -d',' -f3)
            
            if [ "$CONV_STATUS" == "CONVERGED" ]; then
                echo "  ✓ Converged in ${ACTUAL_ITERS} iterations (${ELAPSED}s)"
                echo "${MESH_NAME},${Re},${VELOCITY},${ITERS},${ACTUAL_ITERS},${FINAL_RESIDUAL},CONVERGED,${ELAPSED}s" >> "$LOGFILE"
            else
                echo "  ⚠ NOT CONVERGED after ${ACTUAL_ITERS} iterations (residual: ${FINAL_RESIDUAL})"
                echo "${MESH_NAME},${Re},${VELOCITY},${ITERS},${ACTUAL_ITERS},${FINAL_RESIDUAL},NOT_CONVERGED,${ELAPSED}s" >> "$LOGFILE"
                NOT_CONVERGED_CASES+=("${MESH_NAME}_Re${Re}")
            fi
        else
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "  ✗ FAILED (check results/${MESH_NAME}/Re${Re}/console.log)"
            echo "${MESH_NAME},${Re},${VELOCITY},${ITERS},N/A,N/A,FAILED,${ELAPSED}s" >> "$LOGFILE"
            FAILED_CASES+=("${MESH_NAME}_Re${Re}")
        fi
        
        # Cleanup journal file (optional)
        # rm -f "run_${MESH_NAME}_Re${Re}.jou"
    done
done

# ===================== SUMMARY =====================
echo ""
echo "=========================================="
echo "           BATCH COMPLETE                 "
echo "=========================================="
echo "Total cases:    $TOTAL_CASES"
echo "Converged:      $((TOTAL_CASES - ${#FAILED_CASES[@]} - ${#NOT_CONVERGED_CASES[@]}))"
echo "Not converged:  ${#NOT_CONVERGED_CASES[@]}"
echo "Failed:         ${#FAILED_CASES[@]}"

if [ ${#NOT_CONVERGED_CASES[@]} -gt 0 ]; then
    echo ""
    echo "⚠ Cases needing attention (not converged):"
    printf '  %s\n' "${NOT_CONVERGED_CASES[@]}"
fi

if [ ${#FAILED_CASES[@]} -gt 0 ]; then
    echo ""
    echo "✗ Failed cases:"
    printf '  %s\n' "${FAILED_CASES[@]}"
fi

echo ""
echo "Results saved to: results/"
echo "Summary log: $LOGFILE"
echo "Batch ended: $(date)" >> "$LOGFILE"
