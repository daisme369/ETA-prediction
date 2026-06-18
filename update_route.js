const fs = require('fs');
const current = JSON.parse(fs.readFileSync('route.json', 'utf8'));
const newRoute = {
    "success": true,
    "st": true,
    "dt": {
        "FleetID": 999,
        "Enterprise": "Vinbus",
        "Code": "OCT1",
        "Name": "KĐT Royal City - Ocean City",
        "OperationsTime": "0|T2 - CN|06:00,23:59",
        "Frequency": "10-15-30 phút/chuyến",
        "BusCount": "12",
        "Cost": "0VNĐ/lượt",
        "CostInt": 0,
        "FirstStation": "Royal City R1",
        "LastStation": "Công Viên Vin Wonders - Water Park",
        "Go": {
            "Anomaly": 0,
            "Route": "Royal City R1 - Cổng Vũ Môn - Sân vận động Thượng Đình - Trường Chinh - Minh Khai - Times City - KĐT Ocean Park 1 (S2.09, Nhà để xe Hải Đăng, S2.15) - Cầu Sóng The K Town - Kinh đô Ánh Sáng - Công viên VinWonders Wave Park - Công viên VinWonders Water Park.",
            "Geo": [
                { "Lng": 105.815186, "Lat": 20.999474 },
                { "Lng": 106.012345, "Lat": 20.898765 }
            ],
            "Station": [
                { "ObjectID": 9001, "Code": "9001", "Name": "Royal City R1", "FleetOver": "OCT1", "Geo": { "Lng": 105.815186, "Lat": 20.999474 } },
                { "ObjectID": 9002, "Code": "9002", "Name": "Cổng Vũ Môn - TTTM", "FleetOver": "OCT1", "Geo": { "Lng": 105.816200, "Lat": 20.998500 } },
                { "ObjectID": 9003, "Code": "9003", "Name": "Sân Vận Động Thượng Đình - 129 Nguyễn Trãi", "FleetOver": "01,02,27,OCT1", "Geo": { "Lng": 105.815186, "Lat": 20.999474 } },
                { "ObjectID": 9004, "Code": "9004", "Name": "Đối Diện 196 Trường Chinh", "FleetOver": "16,19,24,E01,E04,OCT1", "Geo": { "Lng": 105.830223, "Lat": 21.000231 } },
                { "ObjectID": 9005, "Code": "9005", "Name": "Viện Kinh Tế Và Kỹ Thuật Dệt May - 478 Minh Khai", "FleetOver": "19,24,E01,E04,E08,OCT1", "Geo": { "Lng": 105.867065, "Lat": 20.997837 } },
                { "ObjectID": 9006, "Code": "9006", "Name": "Phòng Khám Vinmec - TTTM Times City", "FleetOver": "OCT1", "Geo": { "Lng": 105.868500, "Lat": 20.996100 } },
                { "ObjectID": 9007, "Code": "9007", "Name": "T1 Vinhomes Times City", "FleetOver": "OCT1", "Geo": { "Lng": 105.869200, "Lat": 20.995300 } },
                { "ObjectID": 9008, "Code": "9008", "Name": "Tòa Nhà S2.09 (KĐT Ocean Park 1)", "FleetOver": "OCP1,OCT1", "Geo": { "Lng": 105.940500, "Lat": 20.990100 } },
                { "ObjectID": 9009, "Code": "9009", "Name": "Nhà Để xe Hải Đăng", "FleetOver": "OCP1,OCT1", "Geo": { "Lng": 105.941200, "Lat": 20.989200 } },
                { "ObjectID": 9010, "Code": "9010", "Name": "Tòa Nhà S2.15", "FleetOver": "E01,E03,OCT1", "Geo": { "Lng": 105.943850, "Lat": 20.990816 } },
                { "ObjectID": 9011, "Code": "9011", "Name": "Cầu Sóng The K Town", "FleetOver": "OCT1", "Geo": { "Lng": 105.980100, "Lat": 20.950200 } },
                { "ObjectID": 9012, "Code": "9012", "Name": "Kinh Đô Ánh Sáng", "FleetOver": "OCT1", "Geo": { "Lng": 105.981500, "Lat": 20.949500 } },
                { "ObjectID": 9013, "Code": "9013", "Name": "TTTM Center Point Đại Lộ Ngọc Trai", "FleetOver": "OCT1", "Geo": { "Lng": 105.983000, "Lat": 20.948000 } },
                { "ObjectID": 9014, "Code": "9014", "Name": "Công Viên VinWonders Wave Park", "FleetOver": "OCT1", "Geo": { "Lng": 105.985500, "Lat": 20.945500 } },
                { "ObjectID": 9015, "Code": "9015", "Name": "Công Viên VinWonders Water Park", "FleetOver": "OCT1", "Geo": { "Lng": 106.012345, "Lat": 20.898765 } }
            ]
        },
        "Re": {
            "Anomaly": 0,
            "Route": "Công viên VinWonders Water Park - Venice - Bến trả Ocean Park 2, 3 - Nhà để xe Hải Đăng - Tòa nhà S2.09 - Times City - Trường Chinh - Đại học Y - Cổng Vũ Môn - Royal City R1.",
            "Geo": [
                { "Lng": 106.012345, "Lat": 20.898765 },
                { "Lng": 105.815186, "Lat": 20.999474 }
            ],
            "Station": [
                { "ObjectID": 9015, "Code": "9015", "Name": "Công Viên VinWonders Water Park", "FleetOver": "OCT1", "Geo": { "Lng": 106.012345, "Lat": 20.898765 } },
                { "ObjectID": 9016, "Code": "9016", "Name": "Đối Diện Cổng Grand World - The Venice", "FleetOver": "OCT1", "Geo": { "Lng": 106.011000, "Lat": 20.899000 } },
                { "ObjectID": 9017, "Code": "9017", "Name": "Bến Trả Ocean Park 2, 3", "FleetOver": "OCT1", "Geo": { "Lng": 105.980500, "Lat": 20.950100 } },
                { "ObjectID": 9009, "Code": "9009", "Name": "Nhà Để xe Hải Đăng", "FleetOver": "OCP1,OCT1", "Geo": { "Lng": 105.941200, "Lat": 20.989200 } },
                { "ObjectID": 9008, "Code": "9008", "Name": "Tòa Nhà S2.09 (KĐT Ocean Park 1)", "FleetOver": "OCP1,OCT1", "Geo": { "Lng": 105.940500, "Lat": 20.990100 } },
                { "ObjectID": 9018, "Code": "9018", "Name": "Đối Diện Vinhomes Times City", "FleetOver": "OCT1", "Geo": { "Lng": 105.869000, "Lat": 20.995000 } },
                { "ObjectID": 9019, "Code": "9019", "Name": "Đối Diện Bảo Tàng Phòng Không Không Quân - Trường Chinh", "FleetOver": "16,19,24,E01,E04,OCT1", "Geo": { "Lng": 105.831060, "Lat": 21.000219 } },
                { "ObjectID": 9020, "Code": "9020", "Name": "Đại Học Y Hà Nội (Tôn Thất Tùng)", "FleetOver": "OCT1", "Geo": { "Lng": 105.828500, "Lat": 21.003500 } },
                { "ObjectID": 9021, "Code": "9021", "Name": "Cổng Vũ Môn - TTTM (Chiều Về)", "FleetOver": "OCT1", "Geo": { "Lng": 105.816000, "Lat": 20.998200 } },
                { "ObjectID": 9001, "Code": "9001", "Name": "Royal City R1", "FleetOver": "OCT1", "Geo": { "Lng": 105.815186, "Lat": 20.999474 } }
            ]
        }
    },
    "msg": "Construct"
};
if (Array.isArray(current.dt)) {
    // Avoid adding duplicates if run multiple times
    if (!current.dt.some(r => r.Code === "OCT1")) {
        current.dt.push(newRoute.dt);
    }
} else {
    if (current.dt && current.dt.Code !== "OCT1") {
        current.dt = [current.dt, newRoute.dt];
    } else {
        current.dt = [current.dt];
    }
}
fs.writeFileSync('route.json', JSON.stringify(current, null, 4));
console.log('Successfully updated route.json');
