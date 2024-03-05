clc; clear all; close all;
test = 3;
%% scenario
lla0 = [40 116 50]; % 纬经高
if test == 1
    s = drivingScenario('GeoReference',lla0);
    v = vehicle(s);
    
    waypoints = [-11 -0.25 0;
        -1 -0.25 0;
        -0.6 -0.4 0;
        -0.6 -9.3 0];
    speed = [1.5;0;0.5;1.5];
    smoothTrajectory(v,waypoints,speed);

    figure
    plot(waypoints(:,1),waypoints(:,2),'-o')
    xlabel('X (m)')
    ylabel('Y (m)')
    title('Vehicle Position Waypoints')

elseif test == 2
    s = drivingScenario('GeoReference',lla0);
    v = vehicle(s);
    
    waypoints = [0,0,0;10,0,0];
    speed = [0;2];
    smoothTrajectory(v,waypoints,speed);

    figure
    plot(waypoints(:,1),waypoints(:,2),'-o')
    xlabel('X (m)')
    ylabel('Y (m)')
    title('Vehicle Position Waypoints')

elseif test == 3 
    [s,v] = createDrivingScenario();

end
%% create sensors
mountingLocationIMU = [3 4 5];
mountingAnglesIMU = [0 0 0];
% Convert orientation offset from Euler angles to quaternion.
orientVeh2IMU = quaternion(mountingAnglesIMU,'eulerd','ZYX','frame');
% ReferenceFrame must be specified as ENU.
imu = imuSensor('SampleRate',1/s.SampleTime,'ReferenceFrame','ENU');
[a,w]=imu([1 2 3], [1 2 3], orientVeh2IMU)

% SpecSheet1 = accelparams( ...
%     'MeasurementRange',19.62, ...
%     'Resolution',0.00059875, ...
%     'ConstantBias',0.4905, ...
%     'AxesMisalignment',2, ...
%     'NoiseDensity',0.003924, ...
%     'BiasInstability',0, ...
%     'TemperatureBias', [0.34335 0.34335 0.5886], ...
%     'TemperatureScaleFactor', 0.02);
% imu.Accelerometer = SpecSheet1;
% imu.Accelerometer

mountingLocationGPS = [1 2 3];
mountingAnglesGPS = [50 40 30];
% Convert orientation offset from Euler angles to quaternion.
orientVeh2GPS = quaternion(mountingAnglesGPS,'eulerd','ZYX','frame');
% The GeoReference property in drivingScenario is equivalent to
% the ReferenceLocation property in gpsSensor.
% ReferenceFrame must be specified as ENU.
gps = gpsSensor('ReferenceLocation',lla0,'ReferenceFrame','ENU');

encoder = wheelEncoderAckermann('TrackWidth',v.Width,...
    'WheelBase',v.Wheelbase,'SampleRate',1/s.SampleTime);

%% run simulation
% V readings
posVeh_t = [];
orientVeh_t = [];
velVeh_t = [];
accelVeh_t = [];
angvelVeh_t = [];
% IMU readings.
accel = [];
gyro = [];
% Wheel encoder readings.
ticks = [];
% GPS readings.
lla = [];
gpsVel = [];
% Define the rate of the GPS compared to the simulation rate.
simSamplesPerGPS = (1/s.SampleTime)/gps.SampleRate;
idx = 0;
while advance(s)
    groundTruth = state(v);

    % Unpack the ground truth struct by converting the orientations from
    % Euler angles to quaternions and converting angular velocities form
    % degrees per second to radians per second.
    posVeh = groundTruth.Position;
    orientVeh = quaternion(fliplr(groundTruth.Orientation), 'eulerd', 'ZYX', 'frame');
    velVeh = groundTruth.Velocity;
    accVeh = groundTruth.Acceleration;
    angvelVeh = deg2rad(groundTruth.AngularVelocity);
    
    posVeh_t(end+1,:) = posVeh;
    % orientVeh_t(end+1,:) = orientVeh;
    velVeh_t(end+1,:) = velVeh;
    accelVeh_t(end+1,:) = accVeh;
    angvelVeh_t(end+1,:) = angvelVeh;

    
    % Convert motion quantities from vehicle frame to IMU frame.
    [posIMU,orientIMU,velIMU,accIMU,angvelIMU] = transformMotion( ...
        mountingLocationIMU,orientVeh2IMU, ...
        posVeh,orientVeh,velVeh,accVeh,angvelVeh);
    [a,w] = imu(accIMU,angvelIMU,orientIMU); 
    accel(end+1,:) = -a;
    gyro(end+1,:) = w;

    ticks(end+1,:) = encoder(velVeh, angvelVeh, orientVeh); 
    
    % Only generate a new GPS sample when the simulation has advanced
    % enough.
    if (mod(idx, simSamplesPerGPS) == 0)
        % Convert motion quantities from vehicle frame to GPS frame.
        [posGPS,orientGPS,velGPS,accGPS,angvelGPS] = transformMotion(...
            mountingLocationGPS, orientVeh2GPS,...
            posVeh,orientVeh,velVeh,accVeh,angvelVeh);
        [lla(end+1,:), gpsVel(end+1,:)] = gps(posGPS,velGPS);
    end
    idx = idx + 1;
end

%% visualize
figure
plot(ticks)
ylabel('Wheel Ticks')
title('Wheel Encoder')
figure
plot(accel)
ylabel('m/s^2')
title('Accelerometer')
xlim([0 1000])
ylim([-10.0 5.0])

figure
plot(accelVeh_t)
ylabel('m/s^2')
title('AccelerometerVeh')
xlim([0 1000])
ylim([-10.0 5.0])

figure
plot(gyro)
ylabel('rad/s')
title('Gyroscope')
ylim([-1.0 1.0])
figure
plot(angvelVeh_t)
ylabel('rad/s')
title('GyroscopeVeh')
ylim([-1.0 1.0])

figure
geoplot(lla(:,1),lla(:,2))
title('GPS Position')
figure
plot(gpsVel)
ylabel('m/s')
title('GPS Velocity')
