# CE3K

The CE3K simulator is intended to help with evaluating scheduling techniques in a more comprehensive multi-function, multi-sensor technique.

This is a _continuing environment_ designed for evaluating the performance of scheduling algorithms.
This means it should be able to run without resets.

## Questions for Sunila

- In slides, it looks like search cares about range, is this the case? 

## Questions for Jack


## Assumptions

- There's a bit of a time-warp that happens where sensors realize the result of their dwell at the beginning of the dwell and the sensor is just busy with something arbitrary. This is because I didn't want to keep track of which exact tracker was keeping each sensor busy. This I can just rely on what is in the action for each step and don't need to keep track of what each sensor is actually doing. I think over a large number of steps this is totally meaningless, but good to be aware of if you're stepping through individual trackers and wondering why they seem to know what happened before they did anything, or why when a sensor stops being busy nobody learns anything.

## Backlog

- [ ] drop penalty for functions not serviced fast enough
- [ ] implement 2 sensors (sensors enter busy state)
- [ ] create reference batch of scheduling algorithms
- [ ] implement EST scheduler
- [ ] implement EDF scheduler
- [ ] consider BWW implementation
- [ ] consider Auction-based or QRAM-based solver?
- [ ] measure seconds/megastep on a100, or maybe local 2x1080ti?. Translate to simseconds/wallseconds. Consider CI pipeline to re-run this test to monitor for regressions.
- [ ] two-speed-search: Search beams on horizon should occupy about 50% of search time. Longer dwells? Two search zones? Two search functions? Let's not blow up action space too much. If we've got two sensors
- [ ] brainstorm ways to visualize queue, especially with 100+ running trackers
- [ ] create visualization of 3d environment
- [ ] create kalman filter based measurements of actual state
- [ ] define alternative reward function based on belief estimate

## Platform

This represents, for example, a ship.

- **rock:** This is a point at the origin with no kinematics.
- todo: Guided Missile Destroyer - 50km/hr, 4 sky-7 radars providing a 360 view,
- todo: Implement some sort of "Sensor Management System" to combine all sensors together.

### Sensors

- mounting location
- sky-7 radar: s-band active electronically scanned array
- todo: multi-target tracker, we need the whole Observations->Assignment->Track pipeline, with tracks evolving from initialization, confirmation, update, lost, and deletion
- todo: nato link16/22

## Precision

So I started building this hoping to keep everything in 16 bits of precision.
Considering the very tiny time scales we're incrementing over, this seems more than adequate, and yet, mostly just due to the gigantic search area.
I'd like to come up with some clever way to handle the need to have precise position updates that can be sub-millisecond, and yet in theory cover the entire 200km+ search range. Maybe I need to break the universe up into chunks? That said, it's easier to just use 32bits of precision than it is to try to dream up some custom encoding.
