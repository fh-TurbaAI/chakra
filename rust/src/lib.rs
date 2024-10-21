use pyo3::prelude::*;

use pyo3::types::PyList;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone)]
struct RSKinetoOperator {
    #[pyo3(get, set)]
    id: i32,
    #[pyo3(get, set)]
    inclusive_dur: f64,
    #[pyo3(get, set)]
    exclusive_dur: f64,
    #[pyo3(get, set)]
    timestamp: f64,
    #[pyo3(get, set)]
    rf_id: f64,
    #[pyo3(get, set)]
    name: String
}

#[pymethods]
impl RSKinetoOperator {
    #[new]
    fn new(id: i32,
           inclusive_dur: f64,
           exclusive_dur: f64,
           timestamp: f64,
           rf_id: f64,
           name: String) -> Self {
        RSKinetoOperator {
            id,
            inclusive_dur,
            exclusive_dur,
            timestamp,
            rf_id,
            name
        }
    }
}

#[pyclass]
struct DurationCalculator;

#[pymethods]
impl DurationCalculator {
    pub fn calculate_exclusive_dur_rs(&self, kineto_ops: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
        // Convert PyList to Vec<KinetoOperator>
        let sorted_ops: Vec<RSKinetoOperator> = kineto_ops.extract()?;

        // Compute exclusive durations in parallel
        let exclusive_durs: Vec<f64> = (0..sorted_ops.len())
            .into_par_iter()
            .map(|i| self.get_exclusive_dur_for_op_rs(&sorted_ops, i))
            .collect();

        Ok(exclusive_durs)
    }
    #[new]
    fn new() -> Self {
        DurationCalculator{}
    }
}

impl DurationCalculator {
    pub fn get_exclusive_dur_for_op_rs(&self, operators: &Vec<RSKinetoOperator>, op_index: usize) -> f64 {
        let op = &operators[op_index];
        let mut exclusive_dur = op.inclusive_dur;
        let mut overlapping_regions: Vec<(f64, f64)> = Vec::new();

        // Identify overlapping regions with child operators
        for child_op in &operators[op_index + 1..] {
            if child_op.timestamp >= op.timestamp && (child_op.timestamp + child_op.inclusive_dur) <= (op.timestamp + op.inclusive_dur) {
                let overlap_start = child_op.timestamp;
                let overlap_end = child_op.timestamp + child_op.inclusive_dur;
                overlapping_regions.push((overlap_start, overlap_end));
            }
            if (op.timestamp + op.inclusive_dur) < child_op.timestamp {
                break;
            }
        }

        // Merge overlapping regions and calculate exclusive duration
        let merged_regions = Self::merge_overlapping_intervals(&overlapping_regions);
        for (start, end) in merged_regions.clone() {
            exclusive_dur -= end - start;
        }

        // Check if exclusive_dur is not negative or zero
        if exclusive_dur < 0.0 {
            let error_msg = format!(
                "Exclusive duration calculation error for node '{}'(id:{} (ts: {}, inclusive_dur: {}, rf_id: {}): Duration cannot be less than zero ({}) ({:?}).",
                op.name, op.id, op.timestamp, op.inclusive_dur, op.rf_id, exclusive_dur, merged_regions
            );
            eprintln!("{}", error_msg);
        }

        exclusive_dur
    }

    fn merge_overlapping_intervals(intervals: &[(f64, f64)]) -> Vec<(f64, f64)> {
        if intervals.is_empty() {
            return Vec::new();
        }

        // Sort intervals based on the start time
        let mut sorted_intervals = intervals.to_vec();
        sorted_intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // sorted_intervals.sort_by_key(|&(start, _)| start);

        let mut merged: Vec<(f64, f64)> = Vec::new();
        merged.push(sorted_intervals[0]);

        for &current in &sorted_intervals {
            let prev = *merged.last().unwrap();
            if current.0 <= prev.1 {
                // There is overlap, merge the current interval with the previous one
                if let Some(last) = merged.last_mut() {
                    *last = (prev.0, prev.1.max(current.1));
                }
                // merged.last_mut().unwrap().1 = prev.1.max(current.1);
            } else {
                // No overlap, add the current interval
                merged.push(current);
            }
        }

        merged
    }
}


/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rusty_chakra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_class::<DurationCalculator>()?;
    let _ = m.add_class::<RSKinetoOperator>()?;
    Ok(())
}