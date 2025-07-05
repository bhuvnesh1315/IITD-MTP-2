#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/mobility-helper.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"
#include <iomanip>

using namespace ns3;

std::map<uint16_t, uint32_t> g_cellIdToNodeIdMap;
std::map<uint64_t, uint32_t> g_imsiToNodeIdMap;
std::map<uint16_t, uint32_t> g_rntiToNodeIdMap;
std::map<std::tuple<uint32_t, uint32_t, uint16_t>, double> g_neighborRsrpMap;
std::ofstream g_rsrpTraceFile;

NS_LOG_COMPONENT_DEFINE("3x4gridTraceApplication");


void RsrpReceiveCallback(std::string context, Ptr<const Packet> Pkt, const Address& a, const Address& b)
{   
    std::cout << context << std::endl
              << "\tRxTrace Size: " << Pkt->GetSize() 
              << " From:"<< InetSocketAddress:: ConvertFrom(a).GetIpv4()
              << " Local Address: " << InetSocketAddress::ConvertFrom(b).GetIpv4() << std::endl;
}

void RsrpTransmitCallback(std::string ctx, Ptr<const Packet> Pkt, const Address& a, const Address& b)
{
    std::cout << ctx << std::endl
              << "\tTxTrace Size: " << Pkt->GetSize() 
              << " Local Address: " << InetSocketAddress::ConvertFrom(a).GetIpv4()
              << " Target: " << InetSocketAddress::ConvertFrom(b).GetIpv4() << std::endl;
}

void DirectPhyMeasurement(Ptr<NrHelper> nrHelper, NetDeviceContainer& ueNetDev)
{
    double currentTime = Simulator::Now().GetSeconds();
    
    
    for (uint32_t i = 0; i < ueNetDev.GetN(); ++i)
    {
        Ptr<NrUeNetDevice> ueDev = ueNetDev.Get(i)->GetObject<NrUeNetDevice>();
        if (!ueDev) continue;
        
        uint32_t ueNodeId = ueDev->GetNode()->GetId();
        uint64_t imsi = ueDev->GetImsi();
        
        // Get UE PHY for measurements
        Ptr<NrUePhy> uePhy = nrHelper->GetUePhy(ueNetDev.Get(i), 0);
        if (uePhy)
        {
            double rsrp = uePhy->GetRsrp();
            uint16_t cellId = uePhy->GetCellId();
            
            // Find serving gNB
            uint32_t servingGnbId = 0;
            if (g_cellIdToNodeIdMap.find(cellId) != g_cellIdToNodeIdMap.end())
            {
                servingGnbId = g_cellIdToNodeIdMap[cellId];
            }
            
            // Store measurement
            auto key = std::make_tuple(ueNodeId, servingGnbId, cellId);
            g_neighborRsrpMap[key] = rsrp;
            
            std::cout << "UE " << ueNodeId << " (IMSI: " << imsi << ") - "
                      << "Serving Cell " << cellId << " (gNB " << servingGnbId << ") - "
                      << "RSRP: " << std::fixed << std::setprecision(2) << rsrp << " dBm" << std::endl;
            
            // Write to trace file
            if (g_rsrpTraceFile.is_open())
            {
                g_rsrpTraceFile << currentTime << "\t" << ueNodeId << "\t" << imsi << "\t" 
                               << cellId << "\t" << servingGnbId << "\t" << rsrp << "\t"
                               << "Direct_Serving" << std::endl;
            }
        }
    }
    
    // Schedule next measurement
    Simulator::Schedule(Seconds(1.0), &DirectPhyMeasurement, nrHelper, ueNetDev);
}

// Function to populate RNTI mapping after attachment
void PopulateRntiMapping(const NetDeviceContainer& ueNetDevices)
{
    for (uint32_t i = 0; i < ueNetDevices.GetN(); ++i)
    {
        Ptr<NrUeNetDevice> ueDev = ueNetDevices.Get(i)->GetObject<NrUeNetDevice>();
        if (ueDev)
        {
            Ptr<NrUeRrc> ueRrc = ueDev->GetRrc();
            if (ueRrc)
            {
                uint16_t rnti = ueRrc->GetRnti();
                uint32_t nodeId = ueDev->GetNode()->GetId();
                g_rntiToNodeIdMap[rnti] = nodeId;
                
                // std::cout << "UE " << nodeId << " has RNTI: " << rnti << std::endl;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    // Configuration parameters
    uint16_t gridRows = 3;
    uint16_t gridColumns = 4;
    uint16_t totalgNbs = gridRows * gridColumns;
    uint16_t totalUes = 100;
    bool logging = false;
    bool doubleOperationalBand = true;

    // Traffic parameters
    uint32_t udpPacketSizeULL = 100;
    uint32_t udpPacketSizeBe = 1252;
    uint32_t lambdaULL = 10000;
    uint32_t lambdaBe = 10000;

    // Simulation parameters
    Time simTime = MilliSeconds(1000);
    Time udpAppStartTime = MilliSeconds(400);

    // NR parameters
    // uint16_t numerologyBwp1 = 4;
    double centralFrequencyBand1 = 28e9;
    // double bandwidthBand1 = 50e6;
    uint16_t numerologyBwp2 = 2;
    double centralFrequencyBand2 = 28.2e9;
    double bandwidthBand2 = 50e6;
    double totalTxPower = 35;

    uint16_t numerologyBwp1 = 3;
    double bandwidthBand1 = 60e6;
    
    // Output configuration
    std::string simTag = "3x4gridTraceApplication";
    std::string outputDir = "./";

    CommandLine cmd(__FILE__);
    cmd.AddValue("gridRows", "Number of rows in gNB grid", gridRows);
    cmd.AddValue("gridColumns", "Number of columns in gNB grid", gridColumns);
    cmd.AddValue("totalUes", "Total number of UEs", totalUes);
    cmd.AddValue("logging", "Enable logging", logging);
    cmd.Parse(argc, argv);

    totalgNbs = gridRows * gridColumns;

    if (logging)
    {
        LogComponentEnable("UdpClient", LOG_LEVEL_ALL);
        LogComponentEnable("UdpServer", LOG_LEVEL_ALL);
    }

    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));

    // Create grid scenario
    int64_t randomStream = 1;
    GridScenarioHelper gridScenario;
    gridScenario.SetRows(gridRows);
    gridScenario.SetColumns(gridColumns);
    gridScenario.SetHorizontalBsDistance(30.0);
    gridScenario.SetVerticalBsDistance(30.0);
    gridScenario.SetBsHeight(25);
    int range = 2 - 1.5 + 1;
    int UE_height = rand() % range + 1.5;
    gridScenario.SetUtHeight(UE_height);
    gridScenario.SetSectorization(GridScenarioHelper::SINGLE);
    gridScenario.SetBsNumber(totalgNbs);
    gridScenario.SetUtNumber(totalUes);
    randomStream += gridScenario.AssignStreams(randomStream);
    gridScenario.CreateScenario();

    NS_LOG_INFO("Created " << gridScenario.GetUserTerminals().GetN() << " UEs and "
                           << gridScenario.GetBaseStations().GetN() << " gNBs in a "
                           << gridRows << "x" << gridColumns << " grid");

    
    // Create UE containers
    NodeContainer ueLowLatContainer;
    NodeContainer ueVoiceContainer;

    for (uint32_t j = 0; j < gridScenario.GetUserTerminals().GetN(); ++j)
    {
        Ptr<Node> ue = gridScenario.GetUserTerminals().Get(j);
        if (j % 2 == 0)
        {
            ueLowLatContainer.Add(ue);
        }
        else
        {
            ueVoiceContainer.Add(ue);
        }
    }

    // NR setup
    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(nrEpcHelper);

    // Spectrum configuration
    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;

    CcBwpCreator::SimpleOperationBandConf bandConf1(centralFrequencyBand1,
                                                   bandwidthBand1,
                                                   numCcPerBand);
    OperationBandInfo band1 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf1);

    CcBwpCreator::SimpleOperationBandConf bandConf2(centralFrequencyBand2,
                                                   bandwidthBand2,
                                                   numCcPerBand);
    OperationBandInfo band2 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf2);

    // Channel configuration
    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
    
    if (doubleOperationalBand)
    {
        channelHelper->AssignChannelsToBands({band1, band2});
        allBwps = CcBwpCreator::GetAllBwps({band1, band2});
    }
    else
    {
        channelHelper->AssignChannelsToBands({band1});
        allBwps = CcBwpCreator::GetAllBwps({band1});
    }

    // NR parameters configuration
    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));
    nrEpcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));
    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    uint32_t bwpIdForLowLat = 0;
    uint32_t bwpIdForVoice = 0;
    if (doubleOperationalBand)
    {
        bwpIdForVoice = 1;
        bwpIdForLowLat = 0;
    }

    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB",
                                                 UintegerValue(bwpIdForLowLat));
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(bwpIdForVoice));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(bwpIdForLowLat));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(bwpIdForVoice));

    // Install NR devices
    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(gridScenario.GetBaseStations(), allBwps);
    NetDeviceContainer ueLowLatNetDev = nrHelper->InstallUeDevice(ueLowLatContainer, allBwps);
    NetDeviceContainer ueVoiceNetDev = nrHelper->InstallUeDevice(ueVoiceContainer, allBwps);

    randomStream += nrHelper->AssignStreams(gnbNetDev, randomStream);
    randomStream += nrHelper->AssignStreams(ueLowLatNetDev, randomStream);
    randomStream += nrHelper->AssignStreams(ueVoiceNetDev, randomStream);
    
    MobilityHelper mobility;

    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX",
                                  DoubleValue(5.0),
                                  "MinY",
                                  DoubleValue(5.0),
                                  "DeltaX",
                                  DoubleValue(10.0),
                                  "DeltaY",
                                  DoubleValue(10.0),
                                  "GridWidth",
                                  UintegerValue(11),
                                  "LayoutType",
                                  StringValue("RowFirst"));

    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                              "Bounds",
                              RectangleValue(Rectangle(-50, 100, -50, 100)));
    mobility.Install(ueLowLatContainer);

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(ueVoiceContainer);

    // Configure per-gNB settings
    double totalBandwidth = bandwidthBand1 + bandwidthBand2;
    double x = pow(10, totalTxPower / 10);
    
    for (uint32_t i = 0; i < gnbNetDev.GetN(); ++i)
    {
        // Set Tx power of FBS (last GNB)
        if(i == gnbNetDev.GetN()/2)
        {
            nrHelper->GetGnbPhy(gnbNetDev.Get(i), 0)
            ->SetAttribute("Numerology", UintegerValue(numerologyBwp1));
            
            double power = 10 * log10((bandwidthBand1*0.01 / totalBandwidth) * x);
            
            nrHelper->GetGnbPhy(gnbNetDev.Get(i), 0)
                ->SetAttribute("TxPower", DoubleValue(power));
            std::cout<<"FBS: "<<i<<" txpowerFBS: "<<power<<std::endl;
            continue;
        }

        nrHelper->GetGnbPhy(gnbNetDev.Get(i), 0)
            ->SetAttribute("Numerology", UintegerValue(numerologyBwp1));
        nrHelper->GetGnbPhy(gnbNetDev.Get(i), 0)
            ->SetAttribute("TxPower", DoubleValue(10 * log10((bandwidthBand1 / totalBandwidth) * x)));
        
        
        std::cout<<"txpower Band1: "<<10 * log10((bandwidthBand1 / totalBandwidth) * x)<<std::endl;
        if (doubleOperationalBand)
        {
            nrHelper->GetGnbPhy(gnbNetDev.Get(i), 1)
                ->SetAttribute("Numerology", UintegerValue(numerologyBwp2));
            nrHelper->GetGnbPhy(gnbNetDev.Get(i), 1)
                ->SetTxPower(10 * log10((bandwidthBand2 / totalBandwidth) * x));

            std::cout<<"txpower Band2: "<<10 * log10((bandwidthBand2 / totalBandwidth) * x)<<std::endl;
        }
    }

    // Network setup
    auto [remoteHost, remoteHostIpv4Address] =
        nrEpcHelper->SetupRemoteHost("100Gb/s", 2500, Seconds(0.000));
    InternetStackHelper internet;
    internet.Install(gridScenario.GetUserTerminals());

    Ipv4InterfaceContainer ueLowLatIpIface =
        nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLowLatNetDev));
    Ipv4InterfaceContainer ueVoiceIpIface =
        nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueVoiceNetDev));


    //
std::cout << "\n=== Node Count Analysis ===" << std::endl;
std::cout << "Total nodes in simulation: " << NodeList::GetNNodes() << std::endl;
std::cout << "gNB nodes: " << gridScenario.GetBaseStations().GetN() << std::endl;
std::cout << "UE nodes: " << gridScenario.GetUserTerminals().GetN() << std::endl;
std::cout << "Expected (gNB + UE): " << gridScenario.GetBaseStations().GetN() + 
             gridScenario.GetUserTerminals().GetN() << std::endl;

// Print all node details
for (uint32_t i = 0; i < NodeList::GetNNodes(); ++i) {
    Ptr<Node> node = NodeList::GetNode(i);
    std::cout << "Node " << i << ": ";
    
    // Check if it's a gNB
    bool isGnb = false;
    for (uint32_t j = 0; j < gridScenario.GetBaseStations().GetN(); ++j) {
        if (gridScenario.GetBaseStations().Get(j)->GetId() == i) {
            std::cout << "gNB" << std::endl;
            isGnb = true;
            break;
        }
    }
    
    if (!isGnb) {
        // Check if it's a UE
        bool isUe = false;
        for (uint32_t j = 0; j < gridScenario.GetUserTerminals().GetN(); ++j) {
            if (gridScenario.GetUserTerminals().Get(j)->GetId() == i) {
                std::cout << "UE" << std::endl;
                isUe = true;
                break;
            }
        }
        
        if (!isUe) {
            std::cout << "Infrastructure/EPC node" << std::endl;
        }
    }
}
    // Attach UEs to gNBs
    // nrHelper->AttachToClosestGnb(ueLowLatNetDev, gnbNetDev);
    // nrHelper->AttachToClosestGnb(ueVoiceNetDev, gnbNetDev);

    nrHelper->AttachToMaxRsrpGnb(ueLowLatNetDev, gnbNetDev);
    nrHelper->AttachToMaxRsrpGnb(ueVoiceNetDev, gnbNetDev);

    
    g_rsrpTraceFile.open(outputDir + "/rsrp_trace.txt");
    if (g_rsrpTraceFile.is_open())
    {
        g_rsrpTraceFile << "Time(s)\tUE_NodeID\tRNTI/IMSI\tCellID\tgNB_NodeID\tRSRP(dBm)\tType" << std::endl;
    }

    // Populate global cellId to nodeId map
    for (uint32_t i = 0; i < gnbNetDev.GetN(); ++i)
    {
        Ptr<NrGnbNetDevice> gnbDev = gnbNetDev.Get(i)->GetObject<NrGnbNetDevice>();
        uint16_t cellId = gnbDev->GetCellId();
        uint32_t nodeId = gnbDev->GetNode()->GetId();
        g_cellIdToNodeIdMap[cellId] = nodeId;
        
        std::cout << "gNB " << nodeId << " has Cell ID: " << cellId << std::endl;
    }

    // Populate global IMSI to nodeId map for all UEs
    for (uint32_t i = 0; i < ueLowLatNetDev.GetN(); ++i)
    {
        Ptr<NrUeNetDevice> ueDev = ueLowLatNetDev.Get(i)->GetObject<NrUeNetDevice>();
        uint64_t imsi = ueDev->GetImsi();
        uint32_t nodeId = ueDev->GetNode()->GetId();
        g_imsiToNodeIdMap[imsi] = nodeId;
        
        std::cout << "UE " << nodeId << " has IMSI: " << imsi << std::endl;
    }

    for (uint32_t i = 0; i < ueVoiceNetDev.GetN(); ++i)
    {
        Ptr<NrUeNetDevice> ueDev = ueVoiceNetDev.Get(i)->GetObject<NrUeNetDevice>();
        uint64_t imsi = ueDev->GetImsi();
        uint32_t nodeId = ueDev->GetNode()->GetId();
        g_imsiToNodeIdMap[imsi] = nodeId;
        
        std::cout << "UE " << nodeId << " has IMSI: " << imsi << std::endl;
    }

    Simulator::Schedule(Seconds(0.2), &PopulateRntiMapping, ueLowLatNetDev);
    Simulator::Schedule(Seconds(0.2), &PopulateRntiMapping, ueVoiceNetDev);

    NetDeviceContainer allUeDevices;
    allUeDevices.Add(ueLowLatNetDev);
    allUeDevices.Add(ueVoiceNetDev);
    
    Simulator::Schedule(Seconds(0.5), &DirectPhyMeasurement, nrHelper, allUeDevices);


    // Traffic configuration
    uint16_t dlPortLowLat = 1234;
    uint16_t dlPortVoice = 1235;

    ApplicationContainer serverApps;
    UdpEchoServerHelper dlPacketSinkLowLat(dlPortLowLat);
    UdpEchoServerHelper dlPacketSinkVoice(dlPortVoice);
    serverApps.Add(dlPacketSinkLowLat.Install(ueLowLatContainer));
    serverApps.Add(dlPacketSinkVoice.Install(ueVoiceContainer));

    UdpEchoClientHelper dlClientLowLat(ueLowLatIpIface.GetAddress(0), dlPortLowLat);
    dlClientLowLat.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClientLowLat.SetAttribute("PacketSize", UintegerValue(udpPacketSizeULL));
    dlClientLowLat.SetAttribute("Interval", TimeValue(Seconds(1.0 / lambdaULL)));
    

    NrEpsBearer lowLatBearer(NrEpsBearer::NGBR_LOW_LAT_EMBB);
    Ptr<NrEpcTft> lowLatTft = Create<NrEpcTft>();
    NrEpcTft::PacketFilter dlpfLowLat;
    dlpfLowLat.localPortStart = dlPortLowLat;
    dlpfLowLat.localPortEnd = dlPortLowLat;
    lowLatTft->Add(dlpfLowLat);

    UdpEchoClientHelper dlClientVoice(ueVoiceIpIface.GetAddress(0), dlPortVoice);
    dlClientVoice.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClientVoice.SetAttribute("PacketSize", UintegerValue(udpPacketSizeBe));
    dlClientVoice.SetAttribute("Interval", TimeValue(Seconds(1.0 / lambdaBe)));

    NrEpsBearer voiceBearer(NrEpsBearer::GBR_CONV_VOICE);
    Ptr<NrEpcTft> voiceTft = Create<NrEpcTft>();
    NrEpcTft::PacketFilter dlpfVoice;
    dlpfVoice.localPortStart = dlPortVoice;
    dlpfVoice.localPortEnd = dlPortVoice;
    voiceTft->Add(dlpfVoice);

    // Install applications
    ApplicationContainer clientApps;
    for (uint32_t i = 0; i < ueLowLatContainer.GetN(); ++i)
    {
        Ptr<Node> ue = ueLowLatContainer.Get(i);
        Ptr<NetDevice> ueDevice = ueLowLatNetDev.Get(i);
        Address ueAddress = ueLowLatIpIface.GetAddress(i);
        dlClientLowLat.SetAttribute(
            "Remote",
            AddressValue(addressUtils::ConvertToSocketAddress(ueAddress, dlPortLowLat)));
        clientApps.Add(dlClientLowLat.Install(remoteHost));
        nrHelper->ActivateDedicatedEpsBearer(ueDevice, lowLatBearer, lowLatTft);
    }

    for (uint32_t i = 0; i < ueVoiceContainer.GetN(); ++i)
    {
        Ptr<Node> ue = ueVoiceContainer.Get(i);
        Ptr<NetDevice> ueDevice = ueVoiceNetDev.Get(i);
        Address ueAddress = ueVoiceIpIface.GetAddress(i);
        dlClientVoice.SetAttribute(
            "Remote",
            AddressValue(addressUtils::ConvertToSocketAddress(ueAddress, dlPortVoice)));
        clientApps.Add(dlClientVoice.Install(remoteHost));
        nrHelper->ActivateDedicatedEpsBearer(ueDevice, voiceBearer, voiceTft);
    }

    // Start applications
    serverApps.Start(udpAppStartTime);
    clientApps.Start(udpAppStartTime);
    serverApps.Stop(simTime);
    clientApps.Stop(simTime);

    // Config::Connect("/NodeList/*/ApplicationList/*/$ns3::UdpEchoClient/RxWithAddresses",
    //                 MakeCallback(&RsrpReceiveCallback));

    // Config::Connect("/NodeList/*/ApplicationList/*/$ns3::UdpEchoClient/TxWithAddresses",
    //                 MakeCallback(&RsrpTransmitCallback));
    
    // Flow monitor setup
    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(gridScenario.GetUserTerminals());
    Ptr<ns3::FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);
    monitor->SetAttribute("DelayBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("JitterBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("PacketSizeBinWidth", DoubleValue(20));

    Simulator::Stop(simTime);
    AnimationInterface anim(outputDir + "/" + simTag + ".xml");

    for (uint32_t i = 0; i < gridScenario.GetBaseStations().GetN(); ++i) 
    {
        Ptr<Node> node = gridScenario.GetBaseStations().Get(i);
        anim.UpdateNodeColor(node->GetId(), 0, 255, 0); // Green for gNBs
        anim.UpdateNodeDescription(node->GetId(), "gNB");
        anim.UpdateNodeSize(node->GetId(), 5, 5);
    }

    for (uint32_t i = 0; i < gridScenario.GetUserTerminals().GetN(); ++i) 
    {
        Ptr<Node> node = gridScenario.GetUserTerminals().Get(i);
        anim.UpdateNodeColor(node->GetId(), 255, 0, 0); // red for UEs
        anim.UpdateNodeDescription(node->GetId(), "UE");
        anim.UpdateNodeSize(node->GetId(), 2, 2);
    }
    
    for (uint32_t i = 0; i < 4; i++) 
    {
        Ptr<Node> node = gridScenario.GetUserTerminals().Get(i);
        anim.UpdateNodeColor(totalgNbs+totalUes+i, 0, 0, 255); // Blue for core, etc
        anim.UpdateNodeSize(totalgNbs+totalUes+i, 2, 2);
    }
    
    
    Simulator::Run();

    if (g_rsrpTraceFile.is_open())
    {
        g_rsrpTraceFile.close();
    }

    // Create detailed RSRP results file
    std::ofstream rsrpFile(outputDir + "/rsrp_results_100.txt");
    rsrpFile << "UE_NodeID\tgNB_NodeID\tCellID\tRSRP(dBm)\tMeasurement_Type" << std::endl;

    // Group measurements by UE
    std::map<uint32_t, std::vector<std::tuple<uint32_t, uint16_t, double>>> ueToMeasurements;

    for (const auto& measurement : g_neighborRsrpMap)
    {
        uint32_t ueNodeId = std::get<0>(measurement.first);
        uint32_t gnbNodeId = std::get<1>(measurement.first);
        uint16_t cellId = std::get<2>(measurement.first);
        double rsrp = measurement.second;
        
        ueToMeasurements[ueNodeId].push_back(std::make_tuple(gnbNodeId, cellId, rsrp));
    }

    NS_LOG_INFO("RSRP measurements saved to serving_rsrp_results.txt and rsrp_trace.txt");

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    double averageFlowThroughput = 0.0;
    double averageFlowDelay = 0.0;
    std::string filename = outputDir + "/" + simTag;
    std::ofstream outFile(filename.c_str(), std::ios::out | std::ios::trunc);

    if (!outFile.is_open())
    {
        std::cerr << "Can't open file " << filename << std::endl;
        return 1;
    }

    outFile.setf(std::ios_base::fixed);
    double flowDuration = (simTime - udpAppStartTime).GetSeconds();
    
    for (auto const& stat : stats)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(stat.first);
        std::string proto = (t.protocol == 6) ? "TCP" : (t.protocol == 17) ? "UDP" : "Other";
        
        outFile << "Flow " << stat.first << " (" << t.sourceAddress << ":" << t.sourcePort << " -> "
                << t.destinationAddress << ":" << t.destinationPort << ") proto " << proto << "\n";
        outFile << "  Tx Packets: " << stat.second.txPackets << "\n";
        outFile << "  Tx Bytes:   " << stat.second.txBytes << "\n";
        outFile << "  Rx Bytes:   " << stat.second.rxBytes << "\n";
        
        if (stat.second.rxPackets > 0)
        {
            averageFlowThroughput += stat.second.rxBytes * 8.0 / flowDuration / 1e6;
            averageFlowDelay += 1000 * stat.second.delaySum.GetSeconds() / stat.second.rxPackets;
            
            outFile << "  Throughput: " << stat.second.rxBytes * 8.0 / flowDuration / 1e6 << " Mbps\n";
            outFile << "  Mean delay:  " << 1000 * stat.second.delaySum.GetSeconds() / stat.second.rxPackets << " ms\n";
            outFile << "  Mean jitter: " << 1000 * stat.second.jitterSum.GetSeconds() / stat.second.rxPackets << " ms\n";
        }
        else
        {
            outFile << "  Throughput: 0 Mbps\n";
            outFile << "  Mean delay: 0 ms\n";
            outFile << "  Mean jitter: 0 ms\n";
        }
        outFile << "  Rx Packets: " << stat.second.rxPackets << "\n\n";
    }

    double meanFlowThroughput = averageFlowThroughput / stats.size();
    double meanFlowDelay = averageFlowDelay / stats.size();
    
    outFile << "\nMean flow throughput: " << meanFlowThroughput << " Mbps\n";
    outFile << "Mean flow delay: " << meanFlowDelay << " ms\n";
    outFile.close();

    Simulator::Destroy();
    return 0;
}

